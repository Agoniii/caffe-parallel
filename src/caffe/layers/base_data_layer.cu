#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu_test(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
      (*top)[0]->mutable_gpu_data());
  if (this->output_labels_) {
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
        (*top)[1]->mutable_gpu_data());
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu_root(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top, const int source) {
       switch (this->layer_param_.data_param().backend()){
	case DataParameter_DB_LEVELDB:
	{
	Forward_gpu_test(bottom,top);
	caffe_mpi_send<Dtype>((*top)[0]->mutable_cpu_data(),prefetch_data_.count(),
                source,TAG_DATA_OUT,MPI_COMM_WORLD);
	if (this->output_labels_) {
		caffe_mpi_send<Dtype>((*top)[1]->mutable_cpu_data(),prefetch_label_.count(),
                source,TAG_DATA_OUT_IF,MPI_COMM_WORLD);
	}
	}
	break;
	case DataParameter_DB_LMDB:
	{
	}
	break;
	default:
    LOG(FATAL) << "Unknown database backend";
	}
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
       switch (this->layer_param_.data_param().backend()){
	case DataParameter_DB_LEVELDB:
	{
	MPI_Status status;
	status.MPI_ERROR=0;
	caffe_mpi_recv<Dtype>((*top)[0]->mutable_cpu_data(),prefetch_data_.count(),
                0,TAG_DATA_OUT,MPI_COMM_WORLD,&status);
	DLOG(INFO)<<"Recv Dataout status "<<status.MPI_ERROR;
	caffe_copy(prefetch_data_.count(),(*top)[0]->mutable_cpu_data(),(*top)[0]->mutable_gpu_data());//TODO Is this code needed?
	if (this->output_labels_) {
		caffe_mpi_recv<Dtype>((*top)[1]->mutable_cpu_data(),prefetch_label_.count(),
                0,TAG_DATA_OUT_IF,MPI_COMM_WORLD,&status);
		DLOG(INFO)<<"Recv Dataout status "<<status.MPI_ERROR;
	caffe_copy(prefetch_label_.count(),(*top)[1]->mutable_cpu_data(),(*top)[1]->mutable_gpu_data());//TODO Is this code needed?
	}
	}
	break;
	case DataParameter_DB_LMDB:
	{
	Forward_gpu_test(bottom,top);
	}
	break;
	default:
    LOG(FATAL) << "Unknown database backend";
	}
}
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
