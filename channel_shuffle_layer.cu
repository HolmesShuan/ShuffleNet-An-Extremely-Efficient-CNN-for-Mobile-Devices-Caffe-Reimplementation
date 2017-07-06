#include <algorithm>
#include <vector>

#include "caffe/layers/channel_shuffle_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ChannelShuffleForward(const int n, const Dtype* in, Dtype* out,
    const int bottom_dim, const int feature_dim, const int channels) {
  CUDA_KERNEL_LOOP(index, n) {
    const int n = index / bottom_dim;
    const int i = (index - n*bottom_dim) / feature_dim;
    const int j = index - n*bottom_dim - i*feature_dim;
    const int new_index = n*bottom_dim + j*channels + i; 
    out[new_index] = in[index];
  }
}

template <typename Dtype>
void ChannelShuffleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  if (shuffle_pattern == 0) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    ChannelShuffleForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, bottom_dim_, feature_dim_, channels_);
    CUDA_POST_KERNEL_CHECK;
  } else {
    // TODO(Shuan) : parallelize Forward and Backward as pattern 0
    for (int n = 0; n < num_; n++) {
        for (int g = 0; g < group_; g++) {
            for (int c = 0; c < group_chnl_num_; c+=shf_chnl_num_) {
                int group_index = (c/shf_chnl_num_)%group_;
                int feature_index = (c/shf_chnl_num_)/group_;
                caffe_copy(
                    shf_chnl_num_*feature_dim_,
                    bottom_data+n*bottom_dim_+g*group_dim_+c*feature_dim_,
                    top_data+n*bottom_dim_+group_index*group_dim_+feature_index*feature_dim_*shf_chnl_num_
                );
            }
        }
    }  
  }
  
}

template <typename Dtype>
__global__ void ChannelShuffleBackward(const int n, const Dtype* in, Dtype* out,
    const int bottom_dim, const int feature_dim, const int channels) {
  CUDA_KERNEL_LOOP(index, n) {
    const int n = index / bottom_dim;
    const int j = (index - n*bottom_dim) / channels;
    const int i = index - n*bottom_dim - j*channels;
    const int new_index = n*bottom_dim + j + i*feature_dim; 
    out[new_index] = in[index];
  }
}

template <typename Dtype>
void ChannelShuffleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    if (shuffle_pattern == 0) {
        ChannelShuffleBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, bottom_diff, bottom_dim_, feature_dim_, channels_);
        CUDA_POST_KERNEL_CHECK;
    } else {
       // TODO(Shuan) : parallelize Forward and Backward as pattern 0
       for (int n = 0; n < num_; n++) {
        for (int g = 0; g < group_; g++) {
            for (int c = 0; c < group_chnl_num_; c+=shf_chnl_num_) {
                int group_index = (c/shf_chnl_num_)%group_;
                int feature_index = (c/shf_chnl_num_)/group_;
                caffe_copy(
                    shf_chnl_num_*feature_dim_,
                    top_diff+n*bottom_dim_+group_index*group_dim_+feature_index*feature_dim_*shf_chnl_num_,
                    bottom_diff+n*bottom_dim_+g*group_dim_+c*feature_dim_
                );
            }
        }
      } 
    }
  }
}
INSTANTIATE_LAYER_GPU_FUNCS(ChannelShuffleLayer);
}  // namespace caffe
