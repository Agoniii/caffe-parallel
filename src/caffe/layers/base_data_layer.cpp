#include <string>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()),
      data_transformer_(transform_param_) {
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
}

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  if (top->size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  DataLayerSetUp(bottom, top);
  // The subclasses should setup the datum channels, height and width
  CHECK_GT(datum_channels_, 0);
  CHECK_GT(datum_height_, 0);
  CHECK_GT(datum_width_, 0);
  if (transform_param_.crop_size() > 0) {
    CHECK_GE(datum_height_, transform_param_.crop_size());
    CHECK_GE(datum_width_, transform_param_.crop_size());
  }
  // check if we want to have mean
  if (transform_param_.has_mean_file()) {
    const string& mean_file = transform_param_.mean_file();
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
#if 0
    int fsize=0;
        FILE * fin=NULL;
if(rank==0){
        fin=fopen(mean_file.c_str(),"rb");
        if(fin==NULL)LOG(FATAL)<<"NO this mean file "<< mean_file;//TODO client
        fseek(fin,0,SEEK_END);
        fsize=ftell(fin);
        rewind(fin);
}
        MPI_Bcast(&fsize,1,MPI_INT,0,MPI_COMM_WORLD);
        uint8_t *mean_buffer=(uint8_t*)malloc(fsize);
if(rank==0){
        fread(mean_buffer,fsize,1,fin);
        fclose(fin);
}
        MPI_Bcast(mean_buffer,fsize,MPI_CHAR,0,MPI_COMM_WORLD);
        CodedInputStream* coded_input = new CodedInputStream(mean_buffer,fsize);
        coded_input->SetTotalBytesLimit(1073741824, 536870912);

        blob_proto.ParseFromCodedStream(coded_input);

        delete coded_input;
        free(mean_buffer);
#else
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
#endif
    data_mean_.FromProto(blob_proto);
    CHECK_GE(data_mean_.num(), 1);
    CHECK_GE(data_mean_.channels(), datum_channels_);
    CHECK_GE(data_mean_.height(), datum_height_);
    CHECK_GE(data_mean_.width(), datum_width_);
  } else {
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, datum_channels_, datum_height_, datum_width_);
  }
  mean_ = data_mean_.cpu_data();
  data_transformer_.InitRand();
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  this->prefetch_data_.mutable_cpu_data();
  if (this->output_labels_) {
    this->prefetch_label_.mutable_cpu_data();
  }
  DLOG(INFO) << "Initializing prefetch";
  switch (this->layer_param_.data_param().backend()){
	  case DataParameter_DB_LEVELDB:
		  {
			  if(rank==0){
				  this->CreatePrefetchThread();
				  DLOG(INFO) << "Prefetch initialized.";
			  }
		  }
		  break;
	  case DataParameter_DB_LMDB:
		  {
			  //////this->CreatePrefetchThread();
		  }
		  break;
	  default:
		  LOG(FATAL) << "Unknown database backend";
  }
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::CreatePrefetchThread() {
  this->phase_ = Caffe::phase();
  this->data_transformer_.InitRand();
  CHECK(StartInternalThread()) << "Thread execution failed";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu_test(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
             (*top)[0]->mutable_cpu_data());
  if (this->output_labels_) {
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
               (*top)[1]->mutable_cpu_data());
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu_root(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top,const int source) {
       switch (this->layer_param_.data_param().backend()){
        case DataParameter_DB_LEVELDB:
        {
	Forward_cpu_test(bottom,top);
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
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
       switch (this->layer_param_.data_param().backend()){
        case DataParameter_DB_LEVELDB:
        {
	MPI_Status status;
        status.MPI_ERROR=0;
	caffe_mpi_recv<Dtype>((*top)[0]->mutable_cpu_data(),prefetch_data_.count(),
                0,TAG_DATA_OUT,MPI_COMM_WORLD,&status);
	DLOG(INFO)<<"Recv Dataout status "<<status.MPI_ERROR;
	if (this->output_labels_) {
		caffe_mpi_recv<Dtype>((*top)[1]->mutable_cpu_data(),prefetch_label_.count(),
                0,TAG_DATA_OUT_IF,MPI_COMM_WORLD,&status);
	DLOG(INFO)<<"Recv Dataout if status "<<status.MPI_ERROR;
	}
}
        break;
        case DataParameter_DB_LMDB:
        {
        Forward_cpu_test(bottom,top);
        }
        break;
        default:
    LOG(FATAL) << "Unknown database backend";
        }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
template <typename Dtype> \
void BasePrefetchingDataLayer<Dtype>::Forward_gpu_test(const vector<Blob<Dtype>*>& bottom, \
		     vector<Blob<Dtype>*>* top) { NO_GPU; }
template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu_root(const vector<Blob<Dtype>*>& bottom,
		      vector<Blob<Dtype>*>* top,const int source) { NO_GPU;}
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
