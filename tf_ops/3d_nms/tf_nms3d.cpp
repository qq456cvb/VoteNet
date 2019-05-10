#include <cstdio>
#include <ctime>
#include <cstring> // memset
#include <cstdlib> // rand, RAND_MAX
#include <cmath> // sqrtf
#include <functional>
#include <queue>
#include <vector>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("NonMaxSuppression3D")
    .Attr("iou_threshold: float")
    .Input("bbox: float32")
    .Input("scores: float32")
    .Input("objectiveness: float32")
    .Output("idx: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims; // batch_size * nbbox * 6
        c->WithRank(c->input(0), 3, &dims);
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({::tensorflow::shape_inference::InferenceContext::kUnknownDim, 2});
        c->set_output(0, output);
        return Status::OK();
    });


// Return intersection-over-union overlap between boxes i and j
static inline float IOUGreaterThanThreshold(
    typename TTypes<float, 3>::ConstTensor boxes, int b, int i, int j,
    float iou_threshold) {
    const float xmin_i = std::min<float>(boxes(b, i, 0), boxes(b, i, 3));
    const float ymin_i = std::min<float>(boxes(b, i, 1), boxes(b, i, 4));
    const float zmin_i = std::min<float>(boxes(b, i, 2), boxes(b, i, 5));

    const float xmax_i = std::max<float>(boxes(b, i, 0), boxes(b, i, 3));
    const float ymax_i = std::max<float>(boxes(b, i, 1), boxes(b, i, 4));
    const float zmax_i = std::max<float>(boxes(b, i, 2), boxes(b, i, 5));

    const float xmin_j = std::min<float>(boxes(b, j, 0), boxes(b, j, 3));
    const float ymin_j = std::min<float>(boxes(b, j, 1), boxes(b, j, 4));
    const float zmin_j = std::min<float>(boxes(b, j, 2), boxes(b, j, 5));

    const float xmax_j = std::max<float>(boxes(b, j, 0), boxes(b, j, 3));
    const float ymax_j = std::max<float>(boxes(b, j, 1), boxes(b, j, 4));
    const float zmax_j = std::max<float>(boxes(b, j, 2), boxes(b, j, 5));
    
    const float area_i = (xmax_i - xmin_i) * (ymax_i - ymin_i) * (zmax_i - zmin_i);
    const float area_j = (xmax_j - xmin_j) * (ymax_j - ymin_j) * (zmax_j - zmin_j);
    std::cout << area_i << ", " << area_j << std::endl;
    if (area_i <= 0 || area_j <= 0) return 0.0;
    const float intersection_xmin = std::max<float>(xmin_i, xmin_j);
    const float intersection_ymin = std::max<float>(ymin_i, ymin_j);
    const float intersection_zmin = std::max<float>(zmin_i, zmin_j);
    
    const float intersection_xmax = std::min<float>(xmax_i, xmax_j);
    const float intersection_ymax = std::min<float>(ymax_i, ymax_j);
    const float intersection_zmax = std::min<float>(zmax_i, zmax_j);

    const float intersection_area =
        std::max<float>(intersection_xmax - intersection_xmin, 0.0) *
        std::max<float>(intersection_ymax - intersection_ymin, 0.0) *
        std::max<float>(intersection_zmax - intersection_zmin, 0.0);
    const float iou = intersection_area / (area_i + area_j - intersection_area);
    std::cout << iou << std::endl;
    return iou > iou_threshold;
}


static inline std::function<bool(int, int, int)> CreateIOUSuppressCheckFn(
    const Tensor& boxes, float threshold) {
  typename TTypes<float, 3>::ConstTensor boxes_data = boxes.tensor<float, 3>();
  return std::bind(&IOUGreaterThanThreshold, boxes_data, std::placeholders::_1,
                   std::placeholders::_2, std::placeholders::_3, threshold);
}


void DoNonMaxSuppressionOp(OpKernelContext* context, const Tensor& scores, const Tensor& objectiveness,
                            int batch_size, int num_boxes,
                           const float score_threshold,
                           std::function<bool(int, int, int)> suppress_check_fn) {
//    std::cout << batch_size << ", " << num_boxes << std::endl;
  std::vector<float> scores_data(batch_size * num_boxes);
  std::copy_n(scores.flat<float>().data(), batch_size * num_boxes, scores_data.begin());

  std::vector<float> objective_data(batch_size * num_boxes * 2);
  std::copy_n(objectiveness.flat<float>().data(), batch_size * num_boxes * 2, objective_data.begin());


  // Data structure for selection candidate in NMS.
  struct Candidate {
    int batch_index;
    int box_index;
    float score;
  };

  auto cmp = [](const Candidate bs_i, const Candidate bs_j) {
    return bs_i.score < bs_j.score;
  };
  std::priority_queue<Candidate, std::deque<Candidate>, decltype(cmp)>
      candidate_priority_queue(cmp);
//   std::cout << objective_data.size() << std::endl;
  for (int i = 0; i < scores_data.size(); ++i) {
    if (objective_data[i * 2 + 1] > 0.5) { // object exist
      candidate_priority_queue.emplace(Candidate({i / num_boxes, i % num_boxes, scores_data[i]}));
    }
  }

  std::vector<int> selected;
  std::vector<float> selected_scores;
  Candidate next_candidate;

  while (!candidate_priority_queue.empty()) {
    next_candidate = candidate_priority_queue.top();
    candidate_priority_queue.pop();

    // Overlapping boxes are likely to have similar scores,
    // therefore we iterate through the previously selected boxes backwards
    // in order to see if `next_candidate` should be suppressed.
    bool should_select = true;
    for (int j = selected.size() - 2; j >= 0; j -= 2) {
      if (selected[j * 2] == next_candidate.batch_index && suppress_check_fn(next_candidate.batch_index, next_candidate.box_index, selected[j * 2 + 1])) {
        should_select = false;
        break;
      }
    }

    if (should_select) {
      selected.push_back(next_candidate.batch_index);
      selected.push_back(next_candidate.box_index);
      selected_scores.push_back(next_candidate.score);
    }
  }

  // Allocate output tensors
  Tensor* output_indices = nullptr;
  TensorShape output_shape({static_cast<int>(selected.size() / 2), 2});
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, output_shape, &output_indices));
  TTypes<int, 2>::Tensor output_indices_data = output_indices->tensor<int, 2>();
  std::copy_n(selected.begin(), selected.size(), output_indices_data.data());
}



template <typename Device>
class NonMaxSuppression3DOp : public OpKernel {
 public:
        explicit NonMaxSuppression3DOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("iou_threshold", &threshold_));
            OP_REQUIRES(context, threshold_ >= 0 && threshold_ <= 1, errors::InvalidArgument("3D NMS expects positive threshold between [0, 1]"));

        }

        void Compute(OpKernelContext* context) override {
            const Tensor& bbox_tensor = context->input(0);
            OP_REQUIRES(context, bbox_tensor.dims()==3 && bbox_tensor.shape().dim_size(2)==6, errors::InvalidArgument("3D NMS expects (batch_size, nbbox, 6) bbox shape."));
            int b = bbox_tensor.shape().dim_size(0);
            int n = bbox_tensor.shape().dim_size(1);

            const Tensor& scores_tensor = context->input(1);
            OP_REQUIRES(context, scores_tensor.dims()==2, errors::InvalidArgument("3D NMS expects (batch_size, nbbox) scores shape."));

            const Tensor& objectiveness_tensor = context->input(2);
            OP_REQUIRES(context, objectiveness_tensor.dims()==3 && objectiveness_tensor.shape().dim_size(2)==2, errors::InvalidArgument("3D NMS expects (batch_size, nbbox, 2) objectiveness shape."));

            auto suppress_check_fn = CreateIOUSuppressCheckFn(bbox_tensor, threshold_);
            DoNonMaxSuppressionOp(context, scores_tensor, objectiveness_tensor, b, n,
                          threshold_, suppress_check_fn);
        }
    private:
        float threshold_;
};


REGISTER_KERNEL_BUILDER(Name("NonMaxSuppression3D").Device(DEVICE_CPU), NonMaxSuppression3DOp<CPUDevice>);
