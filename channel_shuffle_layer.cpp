#include <algorithm>
#include <vector>

#include "caffe/layers/channel_shuffle_layer.hpp"

namespace caffe {

template <typename Dtype>
void ChannelShuffleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  shuffle_pattern = this->layer_param_.channel_shuffle_param().shuffle_pattern();
  CHECK(shuffle_pattern == 1 || shuffle_pattern == 0);
  
  channels_ = bottom[0]->channels();
  bottom_dim_ = bottom[0]->count(1);
  num_ = bottom[0]->num();
  feature_dim_ = bottom[0]->count(2);
  
  if (shuffle_pattern == 1) {
    group_ = this->layer_param_.channel_shuffle_param().group();
    shf_chnl_num_ = this->layer_param_.channel_shuffle_param().shuffle_channel_num();
    group_chnl_num_ = channels_/group_;
    group_dim_ = group_chnl_num_*feature_dim_;
    CHECK(group_chnl_num_%group_ == 0);
    CHECK((group_chnl_num_/group_)%shf_chnl_num_ == 0);
  }
  
  // CHECK_EQ(bottom_dim_*num_, bottom[0]->count());
  // CHECK_EQ(feature_dim_, bottom[0]->height()*bottom[0]->width());
}

template <typename Dtype>
void ChannelShuffleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // top[0]->Reshape(bottom[0]->num(), bottom[0]->height(), bottom[0]->width(), bottom[0]->channels());
    top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void ChannelShuffleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  if (shuffle_pattern == 0) {
    // From NxCxHxW to NxHxWxC
    // Our implementation is extremely unfriendly to memory access. 
    // Hopefully, you may leave a hint to us.
    for (int n = 0; n < num_; n++) {
        for (int j = 0; j < feature_dim_; j++) {
            for (int i = 0; i < channels_; i++) {
                *(top_data+n*bottom_dim_+j*channels_+i) = 
                    *(bottom_data+n*bottom_dim_+j+i*feature_dim_); 
            }
        }
    }  
  } else {
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
void ChannelShuffleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* top_diff = top[0]->cpu_diff();
  if (propagate_down[0]) {
    if (shuffle_pattern == 0) {
        // From NxHxWxC to NxCxHxW
        for (int n = 0; n < num_; n++) {
            for (int j = 0; j < feature_dim_; j++) {
                for (int i = 0; i < channels_; i++) {
                    *(bottom_diff + n*bottom_dim_+j+i*feature_dim_) = 
                                    *(top_diff+n*bottom_dim_+j*channels_+i);
                }
            }
        }
    } else {
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

#ifdef CPU_ONLY
STUB_GPU(ChannelShuffleLayer);
#endif

INSTANTIATE_CLASS(ChannelShuffleLayer);
REGISTER_LAYER_CLASS(ChannelShuffle);

}  // namespace caffe
