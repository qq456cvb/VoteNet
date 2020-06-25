#include <cstdio>
#include <ctime>
#include <cstring> // memset
#include <cstdlib> // rand, RAND_MAX
#include <cmath>   // sqrtf
#include <functional>
#include <queue>
#include <vector>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
class Point2f
{
public:
  float x;
  float z;

  Point2f(float xx, float zz) : x(xx), z(zz) {}
};

REGISTER_OP("NonMaxSuppression3D")
    .Input("bbox: float32")
    .Input("scores: float32")
    .Input("objectiveness: float32")
    .Input("iou_threshold: float32")
    .Output("idx: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
      ::tensorflow::shape_inference::ShapeHandle dims; // batch_size * nbbox * 8 * 3
      c->WithRank(c->input(0), 3, &dims);
      ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({::tensorflow::shape_inference::InferenceContext::kUnknownDim, 2});
      c->set_output(0, output);
      return Status::OK();
    });

// 8 * 3
// l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, x
// h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2, y
// w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, z
static inline float area2d(const float* bbox) {
  return sqrtf((bbox[0 * 3] - bbox[1 * 3]) * (bbox[0 * 3] - bbox[1 * 3]) + (bbox[0 * 3 + 2] - bbox[1 * 3 + 2]) * (bbox[0 * 3 + 2] - bbox[1 * 3 + 2]))
    * sqrtf((bbox[1 * 3] - bbox[2 * 3]) * (bbox[1 * 3] - bbox[2 * 3]) + (bbox[1 * 3 + 2] - bbox[2 * 3 + 2]) * (bbox[1 * 3 + 2] - bbox[2 * 3 + 2]));
}

static inline float area3d(const float* bbox) {
  return area2d(bbox) * (bbox[0 * 3 + 1] - bbox[4 * 3 + 1]);
}

// https://www.swtestacademy.com/intersection-convex-polygons-algorithm/
static inline float pointInPolygon(const Point2f &p, const float *poly)
{ // poly : 8 * 3, we need the first 4
  int i;
  int j;
  bool result = false;
  for (i = 0, j = 3; i < 4; j = i++)
  {
    if ((poly[i * 3 + 2] > p.z) != (poly[j * 3 + 2] > p.z) &&
        (p.x < (poly[j * 3] - poly[i * 3]) * (p.z - poly[i * 3 + 2]) / (poly[j * 3 + 2] - poly[i * 3 + 2]) + poly[i * 3]))
    {
      result = !result;
    }
  }
  return result;
}

static inline Point2f *getIntersectionPoint(const Point2f &l1p1, const Point2f &l1p2, const Point2f &l2p1, const Point2f &l2p2)
{
  // std::cout << "l1p1: " << l1p1.x << ", " << l1p1.z << "; l1p2: " << l1p2.x << ", " << l1p2.z<< std::endl;
  // std::cout << "l2p1: " << l2p1.x << ", " << l2p1.z << "; l2p2: " << l2p2.x << ", " << l2p2.z<< std::endl;
  double A1 = l1p2.z - l1p1.z;
  double B1 = l1p1.x - l1p2.x;
  double C1 = A1 * l1p1.x + B1 * l1p1.z;

  double A2 = l2p2.z - l2p1.z;
  double B2 = l2p1.x - l2p2.x;
  double C2 = A2 * l2p1.x + B2 * l2p1.z;

  // std::cout << A1 << ", " << B2 << ", " << A2 << ", " << B1 << std::endl;
  //lines are parallel
  double det = A1 * B2 - A2 * B1;
  if (std::abs(det) < 1e-7)
  {
    // std::cout << "parallel" << std::endl;
    return nullptr; //parallel lines
  }
  else
  {
    double x = (B2 * C1 - B1 * C2) / det;
    double z = (A1 * C2 - A2 * C1) / det;
    bool online1 = ((MIN(l1p1.x, l1p2.x) <= x) && (MAX(l1p1.x, l1p2.x) >= x) && (MIN(l1p1.z, l1p2.z) <= z) && (MAX(l1p1.z, l1p2.z) >= z));
    bool online2 = ((MIN(l2p1.x, l2p2.x) <= x) && (MAX(l2p1.x, l2p2.x) >= x) && (MIN(l2p1.z, l2p2.z) <= z) && (MAX(l2p1.z, l2p2.z) >= z));

    if (online1 && online2)
      return new Point2f(x, z);
  }
  return nullptr; //intersection is at out of at least one segment.
}

static inline std::vector<Point2f> getIntersectionPoints(const Point2f &l1p1, const Point2f &l1p2, const float *poly)
{
  std::vector<Point2f> intersectionPoints;
  for (int i = 0; i < 4; i++)
  {

    int next = (i + 1 == 4) ? 0 : i + 1;

    
    Point2f *ip = getIntersectionPoint(l1p1, l1p2, Point2f(poly[i * 3], poly[i * 3 + 2]), Point2f(poly[next * 3], poly[next * 3 + 2]));

    if (ip != nullptr)
    {
      intersectionPoints.emplace_back(ip->x, ip->z);
      delete ip;
    }
  }
  return intersectionPoints;
}

static inline float intersection(const float *bbox1, const float *bbox2)
{
  std::vector<Point2f> clippedCorners;

  //Add  the corners of poly1 which are inside poly2
  for (int i = 0; i < 4; i++)
  {
    if (pointInPolygon(Point2f(bbox1[i * 3], bbox1[i * 3 + 2]), bbox2))
      clippedCorners.emplace_back(bbox1[i * 3], bbox1[i * 3 + 2]);
  }

  //Add the corners of poly2 which are inside poly1
  for (int i = 0; i < 4; i++)
  {
    if (pointInPolygon(Point2f(bbox2[i * 3], bbox2[i * 3 + 2]), bbox1))
      clippedCorners.emplace_back(bbox2[i * 3], bbox2[i * 3 + 2]);
  }

  //Add  the intersection points
  for (int i = 0, next = 1; i < 4; i++, next = (i + 1 == 4) ? 0 : i + 1)
  {
    auto intersectionPoints = getIntersectionPoints(Point2f(bbox1[i * 3],
                                                            bbox1[i * 3 + 2]),
                                                    Point2f(bbox1[next * 3], bbox1[next * 3 + 2]), bbox2);
    // std::cout << "p1: " << bbox1[i * 3] << ", " << bbox1[i * 3 + 2] << "; p2: " << bbox1[next * 3] << ", " << bbox1[next * 3 + 2]<< std::endl;
    for (const auto &p : intersectionPoints)
    {
    // std::cout << "add intersection" << std::endl;
      clippedCorners.emplace_back(p.x, p.z);
    }
  }

  float mx = 0;
  float mz = 0;
  for (const auto &p : clippedCorners)
  {
    mx += p.x;
    mz += p.z;
  }
  mx /= clippedCorners.size();
  mz /= clippedCorners.size();

  std::sort(clippedCorners.begin(), clippedCorners.end(), [&](const Point2f &p1, const Point2f &p2) {
    return atan2f(p1.z - mz, p1.x - mx) < atan2f(p2.z - mz, p2.x - mx);
  });

  int i, j;
  float area = 0;
  for (i = 0, j = clippedCorners.size() - 1; i < clippedCorners.size(); j = i++)
  {
    area += fabsf((mx * (clippedCorners[i].z - clippedCorners[j].z) + clippedCorners[i].x * (clippedCorners[j].z - mz) + clippedCorners[j].x * (mz - clippedCorners[i].z)) / 2);
  }
  return area;
}

// Return intersection-over-union overlap between boxes i and j
static inline float IOUGreaterThanThreshold(
    typename TTypes<float, 4>::ConstTensor boxes, int b, int i, int j,
    int nboxes, float iou_threshold)
{
  auto box_i_ptr = boxes.data() + b * nboxes * 8 * 3 + i * 8 * 3;
  auto box_j_ptr = boxes.data() + b * nboxes * 8 * 3 + j * 8 * 3;

  float intersection2d = intersection(box_i_ptr, box_j_ptr);
  float iou2d = intersection2d / (area2d(box_i_ptr) + area2d(box_j_ptr) - intersection2d);
  float intersection3d = MAX(MIN(box_i_ptr[1], box_j_ptr[1]) - MAX(box_i_ptr[4 * 3 + 1], box_j_ptr[4 * 3 + 1]), 0) * intersection2d;
  float iou = intersection3d / (area3d(box_i_ptr) + area3d(box_j_ptr) - intersection3d);
  // std::cout << area2d(box_i_ptr) << ", " << area2d(box_j_ptr) << std::endl;
//  std::cout << intersection2d << ", " << iou << std::endl;
  return iou > iou_threshold;
}

static inline std::function<bool(int, int, int)> CreateIOUSuppressCheckFn(
    const Tensor &boxes, float threshold, int nboxes)
{
  typename TTypes<float, 4>::ConstTensor boxes_data = boxes.tensor<float, 4>();
  return std::bind(&IOUGreaterThanThreshold, boxes_data, std::placeholders::_1,
                   std::placeholders::_2, std::placeholders::_3, nboxes, threshold);
}

void DoNonMaxSuppressionOp(OpKernelContext *context, const Tensor &scores, const Tensor &objectiveness,
                           int batch_size, int num_boxes,
                           const float score_threshold,
                           std::function<bool(int, int, int)> suppress_check_fn)
{
  //    std::cout << batch_size << ", " << num_boxes << std::endl;
  std::vector<float> scores_data(batch_size * num_boxes);
  std::copy_n(scores.flat<float>().data(), batch_size * num_boxes, scores_data.begin());

  std::vector<float> objective_data(batch_size * num_boxes * 2);
  std::copy_n(objectiveness.flat<float>().data(), batch_size * num_boxes * 2, objective_data.begin());

  // Data structure for selection candidate in NMS.
  struct Candidate
  {
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
  for (int i = 0; i < scores_data.size(); ++i)
  {
    if (objective_data[i * 2 + 1] > objective_data[i * 2])
    { // object exist
      candidate_priority_queue.emplace(Candidate({i / num_boxes, i % num_boxes, scores_data[i]}));
    }
  }

  std::vector<int> selected;
  std::vector<float> selected_scores;
  Candidate next_candidate;

  while (!candidate_priority_queue.empty())
  {
    next_candidate = candidate_priority_queue.top();

    // Overlapping boxes are likely to have similar scores,
    // therefore we iterate through the previously selected boxes backwards
    // in order to see if `next_candidate` should be suppressed.
    bool should_select = true;
    for (int j = selected.size() - 2; j >= 0; j -= 2)
    {
      if (selected[j] == next_candidate.batch_index && suppress_check_fn(next_candidate.batch_index, next_candidate.box_index, selected[j + 1]))
      {
        should_select = false;
        break;
      }
    }

    if (should_select)
    {
      selected.push_back(next_candidate.batch_index);
      selected.push_back(next_candidate.box_index);
      selected_scores.push_back(next_candidate.score);
    }
    candidate_priority_queue.pop();
  }

  // Allocate output tensors
  Tensor *output_indices = nullptr;
  TensorShape output_shape({static_cast<int>(selected.size() / 2), 2});
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, output_shape, &output_indices));
  TTypes<int, 2>::Tensor output_indices_data = output_indices->tensor<int, 2>();
  std::copy_n(selected.begin(), selected.size(), output_indices_data.data());
}

template <typename Device>
class NonMaxSuppression3DOp : public OpKernel
{
public:
  explicit NonMaxSuppression3DOp(OpKernelConstruction *context) : OpKernel(context)
  {
  }

  void Compute(OpKernelContext *context) override
  {
    const Tensor &bbox_tensor = context->input(0);
    //            std::cout << bbox_tensor.dims() << ", " << bbox_tensor.shape().dim_size(2) << std::endl;
    OP_REQUIRES(context, bbox_tensor.dims() == 4 && bbox_tensor.shape().dim_size(2) == 8 && bbox_tensor.shape().dim_size(3) == 3, errors::InvalidArgument("3D NMS expects (batch_size, nbbox, 8, 3) bbox shape."));
    int b = bbox_tensor.shape().dim_size(0);
    int n = bbox_tensor.shape().dim_size(1);

    const Tensor &scores_tensor = context->input(1);
    OP_REQUIRES(context, scores_tensor.dims() == 2 && scores_tensor.shape().dim_size(0) == b && scores_tensor.shape().dim_size(1) == n, errors::InvalidArgument("3D NMS expects (batch_size, nbbox) scores shape."));

    const Tensor &objectiveness_tensor = context->input(2);
    OP_REQUIRES(context, objectiveness_tensor.dims() == 3 && objectiveness_tensor.shape().dim_size(0) == b && objectiveness_tensor.shape().dim_size(1) == n && objectiveness_tensor.shape().dim_size(2) == 2, errors::InvalidArgument("3D NMS expects (batch_size, nbbox, 2) objectiveness shape."));

    const Tensor &iou_threshold = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(iou_threshold.shape()), errors::InvalidArgument("3D NMS expects scalar threshold"));
    const float iou_threshold_val = iou_threshold.scalar<float>()();
    OP_REQUIRES(context, iou_threshold_val >= 0 && iou_threshold_val <= 1, errors::InvalidArgument("iou_threshold must be in [0, 1]"));

    auto suppress_check_fn = CreateIOUSuppressCheckFn(bbox_tensor, iou_threshold_val, n);
    DoNonMaxSuppressionOp(context, scores_tensor, objectiveness_tensor, b, n,
                          iou_threshold_val, suppress_check_fn);
  }
};

REGISTER_KERNEL_BUILDER(Name("NonMaxSuppression3D").Device(DEVICE_CPU), NonMaxSuppression3DOp<CPUDevice>);
