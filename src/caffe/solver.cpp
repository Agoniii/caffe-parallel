#include <cstdio>

#include <algorithm>
#include <string>
#include <vector>
#include <queue>
#include <sys/time.h>

#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <chrono>
//using namespace std;

#include "caffe/util/semaphores.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "leveldb/db.h"
#include "lmdb.h"

using namespace std;
//TODO need package (start)
int currentUpdateCount=0;
atomic_int undoneIter{0};
condition_variable condUpdateNet;
mutex mutexUpdateNet;
condition_variable condUpdateWeight;
mutex mutexUpdateWeight;
atomic_ulong gskip{0};
//TODO need package (end)
namespace caffe {
#ifdef ASYNCTRAN
	template <typename Dtype>
		void SlaveReadData(shared_ptr<Net<Dtype> > layer, const int endNum ,semaphore* semRead,semaphore* semNext){
			vector<Blob<Dtype>*>* top = layer->getTopPointer();
			MPI_Status status;
			for(int i=0;i<endNum;++i){
				status.MPI_ERROR=0;
#ifdef DIRECTGPU
				caffe_mpi_recv<Dtype>((*top)[0]->mutable_gpu_data(),(*top)[0]->count(),
						0,TAG_DATA_OUT,MPI_COMM_WORLD,&status);
				DLOG(INFO)<<"Recv Dataout status "<<status.MPI_ERROR;
				if (top->size()>1) {
					caffe_mpi_recv<Dtype>((*top)[1]->mutable_gpu_data(),(*top)[1]->count(),
							0,TAG_DATA_OUT_IF,MPI_COMM_WORLD,&status);
					DLOG(INFO)<<"Recv Dataout status "<<status.MPI_ERROR;
				}
#else
				caffe_mpi_recv<Dtype>((*top)[0]->mutable_cpu_data(),(*top)[0]->count(),
						0,TAG_DATA_OUT,MPI_COMM_WORLD,&status);
				DLOG(INFO)<<"Recv Dataout status "<<status.MPI_ERROR;
				if (top->size()>1) {
					caffe_mpi_recv<Dtype>((*top)[1]->mutable_cpu_data(),(*top)[1]->count(),
							0,TAG_DATA_OUT_IF,MPI_COMM_WORLD,&status);
					DLOG(INFO)<<"Recv Dataout status "<<status.MPI_ERROR;
				}
#endif
				semRead->notify();
				semNext->wait();
			}
			LOG(INFO)<<"Slave read thread finished!";
		}
#endif
	template <typename Dtype>
		void ReadDataFromDBandDistribute(Layer<Dtype>* layer, const int tid, const int endNum, const int childProcessSum){
			int startNum = 0;
			int batchsize = layer->layer_param().data_param().batch_size();
			int skip = (tid-1) * batchsize;
			Blob<Dtype> *prefetch_data_ = new Blob<Dtype>;
			Blob<Dtype> *prefetch_label_ = new Blob<Dtype>;
			/*****leveldb *******/
			shared_ptr<leveldb::Iterator> titer_;
			/*****LMDB **********/
			MDB_cursor* mdb_cursor_;
			MDB_val mdb_key_, mdb_value_;

			bool output_label = layer->getOutputLabel();
			switch (layer->layer_param().data_param().backend()) {
				case DataParameter_DB_LEVELDB:
					layer->getLeveldbIter(titer_);
					titer_->SeekToFirst();
					break;
				case DataParameter_DB_LMDB:
					layer->getMdbCursor( &mdb_cursor_);
					CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
							MDB_SUCCESS) << "mdb_cursor_get failed";
					break;
				default:
					LOG(FATAL) << "Unknown database backend";
			}

			layer->reshapeData(*prefetch_data_, *prefetch_label_);
			const int skip1 = (childProcessSum-1) * batchsize;
			for(int Num = startNum; Num < endNum; ++Num){
				if(Num!= startNum)skip = skip1;
				DBGPRT(LOG(INFO)<<"SKIP START "<< Num <<" " <<tid<<" "<<skip);
#if 1
				while (skip-- > 0) {
					switch (layer->layer_param().data_param().backend()) {
						case DataParameter_DB_LEVELDB:
							titer_->Next();
							if (!titer_->Valid()) {
								titer_->SeekToFirst();
							}
							break;
						case DataParameter_DB_LMDB:
							if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
									!= MDB_SUCCESS) {
								CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
											MDB_FIRST), MDB_SUCCESS);
							}
							break;
						default:
							LOG(FATAL) << "Unknown database backend";
					}
				}

#endif
				DBGPRT(LOG(INFO)<<"SKIP FIN READ START"<<Num<<" "<<tid);
				layer->ReadData(titer_, mdb_cursor_, *prefetch_data_, *prefetch_label_);
				DBGPRT(LOG(INFO)<<"READ FIN SEND START"<<Num<<" "<<tid);
#ifdef DIRECTGPU
				caffe_mpi_send<Dtype>(prefetch_data_->mutable_gpu_data(),prefetch_data_->count(), 
						tid,TAG_DATA_OUT,MPI_COMM_WORLD);
				if(output_label){
					caffe_mpi_send<Dtype>(prefetch_label_->mutable_gpu_data(),prefetch_label_->count(), 
							tid,TAG_DATA_OUT_IF,MPI_COMM_WORLD);
				}
#else
				caffe_mpi_send<Dtype>(prefetch_data_->mutable_cpu_data(),prefetch_data_->count(), 
						tid,TAG_DATA_OUT,MPI_COMM_WORLD);
				if(output_label){
					caffe_mpi_send<Dtype>(prefetch_label_->mutable_cpu_data(),prefetch_label_->count(), 
							tid,TAG_DATA_OUT_IF,MPI_COMM_WORLD);
				}
#endif
				DBGPRT(LOG(INFO)<<"SEND FIN"<<Num<<" "<<tid);
			}
			delete prefetch_data_;
			delete prefetch_label_;
			LOG(INFO)<<"Read database thread out! Thread No."<<tid;
		}
	template <typename Dtype>
		void ComputeValueThreadServer( Solver<Dtype>* layer, const int childProcessSum) {
			while(true){
				if(undoneIter <= 0)break;
				{
					unique_lock<mutex> lk(mutexUpdateWeight);
#if 1
					DBGPRT(LOG(INFO)<<"WAIT "<<currentUpdateCount);
					condUpdateWeight.wait_for(lk,chrono::seconds(layer->param().timeout_sec()),
							[=](){return currentUpdateCount >= childProcessSum;});
#else
					condUpdateWeight.wait(lk,[=](){return currentUpdateCount >= childProcessSum || currentUpdateCount >= undoneIter;});//need test
#endif
					DBGPRT(LOG(INFO)<<"WAIT FIN"<<currentUpdateCount);
					undoneIter -= currentUpdateCount;
					if(currentUpdateCount>0){
						layer->ComputeValueServer();
						DBGPRT(LOG(INFO)<<"CVS FIN"<<currentUpdateCount);
						condUpdateNet.notify_all();

					}
				}
			}
		}
	template <typename Dtype>
		void Solver<Dtype>::ComputeValueServer(){
			//ComputeUpdateValueServerThreadGPU();//gpu
			//ComputeUpdateValueServerThread();//cpu
			ComputeUpdateValue();
			bool ifTest = ((iter_+currentUpdateCount)/param_.test_interval())  > (iter_ /param_.test_interval());
			iter_ += currentUpdateCount;
			if(ifTest)
				TestAll();
			currentUpdateCount=0;
		}
	template <typename Dtype>
		void SGDSolver<Dtype>::ComputeUpdateValue() {
			vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
			vector<float>& net_params_lr = this->net_->params_lr();
			vector<float>& net_params_weight_decay = this->net_->params_weight_decay();
			// get the learning rate
			Dtype rate = GetLearningRate();
			if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
				LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
			}
			Dtype momentum = this->param_.momentum();
			Dtype weight_decay = this->param_.weight_decay();
			string regularization_type = this->param_.regularization_type();
			switch (Caffe::mode()) {
				case Caffe::CPU:
					for(int param_id = 0; param_id < net_params.size(); ++param_id){
						memset(net_params[param_id]->mutable_cpu_diff(),0,sizeof(Dtype)*(net_params[param_id]->count()));
					}
					for(int i=0;i<this->childProcessSum;++i){
						if(this->flagComputeEndNeedUpdate[i]==1){
							for(int param_id = 0; param_id < net_params.size(); ++param_id){
//assert(net_params[param_id]->count()!=this->tempdata[i][param_id]->count());
								caffe_axpy(net_params[param_id]->count(),(Dtype)1,
										this->tempdata[i][param_id]->mutable_cpu_data(),
										net_params[param_id]->mutable_cpu_diff());
							}
						}
					}
					for (int param_id = 0; param_id < net_params.size(); ++param_id) {
						caffe_scal(net_params[param_id]->count(),(Dtype)(1.0/currentUpdateCount),net_params[param_id]->mutable_cpu_diff());
						// Compute the value to history, and then copy them to the blob's diff.
						Dtype local_rate = rate * net_params_lr[param_id];
						Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

						if (local_decay) {
							if (regularization_type == "L2") {
								// add weight decay
								caffe_axpy(net_params[param_id]->count(),
										local_decay,
										net_params[param_id]->cpu_data(),
										net_params[param_id]->mutable_cpu_diff());
							} else if (regularization_type == "L1") {
								caffe_cpu_sign(net_params[param_id]->count(),
										net_params[param_id]->cpu_data(),
										temp_[param_id]->mutable_cpu_data());
								caffe_axpy(net_params[param_id]->count(),
										local_decay,
										temp_[param_id]->cpu_data(),
										net_params[param_id]->mutable_cpu_diff());
							} else {
								LOG(FATAL) << "Unknown regularization type: " << regularization_type;
							}
						}

						caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
								net_params[param_id]->cpu_diff(), momentum,
								history_[param_id]->mutable_cpu_data());
						// copy
						caffe_copy(net_params[param_id]->count(),
								history_[param_id]->cpu_data(),
								net_params[param_id]->mutable_cpu_diff());
					}
					break;
				case Caffe::GPU:
#ifndef CPU_ONLY
					for(int param_id = 0; param_id < net_params.size(); ++param_id){
						cudaMemset(net_params[param_id]->mutable_gpu_diff(),0,sizeof(Dtype)*(net_params[param_id]->count()));
					}
					for(int i=0;i<this->childProcessSum;++i){
						if(this->flagComputeEndNeedUpdate[i]==1){
							for(int param_id = 0; param_id < net_params.size(); ++param_id){
//assert(net_params[param_id]->count()!=this->tempdata[i][param_id]->count());
								caffe_gpu_axpy(net_params[param_id]->count(),(Dtype)1,
										this->tempdata[i][param_id]->mutable_gpu_data(),
										net_params[param_id]->mutable_gpu_diff());
							}
						}
					}
					for (int param_id = 0; param_id < net_params.size(); ++param_id) {
						caffe_gpu_scal(net_params[param_id]->count(),(Dtype)(1.0/currentUpdateCount),net_params[param_id]->mutable_gpu_diff());
						// Compute the value to history, and then copy them to the blob's diff.
						Dtype local_rate = rate * net_params_lr[param_id];
						Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

						if (local_decay) {
							if (regularization_type == "L2") {
								// add weight decay
								caffe_gpu_axpy(net_params[param_id]->count(),
										local_decay,
										net_params[param_id]->gpu_data(),
										net_params[param_id]->mutable_gpu_diff());
							} else if (regularization_type == "L1") {
								caffe_gpu_sign(net_params[param_id]->count(),
										net_params[param_id]->gpu_data(),
										temp_[param_id]->mutable_gpu_data());
								caffe_gpu_axpy(net_params[param_id]->count(),
										local_decay,
										temp_[param_id]->gpu_data(),
										net_params[param_id]->mutable_gpu_diff());
							} else {
								LOG(FATAL) << "Unknown regularization type: " << regularization_type;
							}
						}

						caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
								net_params[param_id]->gpu_diff(), momentum,
								history_[param_id]->mutable_gpu_data());
						// copy
						caffe_copy(net_params[param_id]->count(),
								history_[param_id]->gpu_data(),
								net_params[param_id]->mutable_gpu_diff());
					}
#else
					NO_GPU;
#endif
					break;
				default:
					LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
			}
			this->net_->Update();
			for(int i=0;i<this->childProcessSum;++i){
				if(this->flagComputeEndNeedUpdate[i]==1){
					for (int param_id = 0; param_id < net_params.size(); ++param_id) {
						caffe_copy(net_params[param_id]->count(),
								net_params[param_id]->gpu_data(),
								this->tempdata[i][param_id]->mutable_gpu_data());
					}
				}
			}
		}
	template <typename Dtype>
		void SGDSolver<Dtype>::ComputeUpdateValueServerThread(){
			vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
			vector<float>& net_params_lr = this->net_->params_lr();
			vector<float>& net_params_weight_decay = this->net_->params_weight_decay();
			Dtype rate = GetLearningRate();
			if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
				LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
			}
			Dtype momentum = this->param_.momentum();
			Dtype weight_decay = this->param_.weight_decay();
			string regularization_type = this->param_.regularization_type();
			DBGPRT(LOG(INFO)<<"111");
			for(int param_id = 0; param_id < net_params.size(); ++param_id){
				memset(net_params[param_id]->mutable_cpu_diff(),0,sizeof(Dtype)*(net_params[param_id]->count()));
			}
			DBGPRT(LOG(INFO)<<"222");
			for(int i=0;i<this->childProcessSum;++i){
				if(this->flagComputeEndNeedUpdate[i]==1){
					for(int param_id = 0; param_id < net_params.size(); ++param_id){
						caffe_axpy(net_params[param_id]->count(),(Dtype)1,
								this->tempdata[i][param_id]->mutable_cpu_data(),net_params[param_id]->mutable_cpu_diff());
					}
				}
			}
			DBGPRT(LOG(INFO)<<"333");
			assert(currentUpdateCount==0);
			for (int param_id = 0; param_id < net_params.size(); ++param_id) {
				caffe_scal(net_params[param_id]->count(),(Dtype)(1.0/currentUpdateCount),net_params[param_id]->mutable_cpu_diff());
				Dtype local_rate = rate * net_params_lr[param_id];
				Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
				if (local_decay) {
					if (regularization_type == "L2") {
						caffe_axpy(net_params[param_id]->count(),
								local_decay,
								net_params[param_id]->cpu_data(),
								net_params[param_id]->mutable_cpu_diff());
					} else if (regularization_type == "L1") {
						caffe_cpu_sign(net_params[param_id]->count(),
								net_params[param_id]->cpu_data(),
								temp_[param_id]->mutable_cpu_data());
						caffe_axpy(net_params[param_id]->count(),
								local_decay,
								temp_[param_id]->cpu_data(),
								net_params[param_id]->mutable_cpu_diff());
					} else {
						LOG(FATAL) << "Unknown regularization type: " << regularization_type;
					}
				}

				caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
						net_params[param_id]->cpu_diff(), momentum,
						history_[param_id]->mutable_cpu_data());
				caffe_copy(net_params[param_id]->count(),
						history_[param_id]->cpu_data(),
						net_params[param_id]->mutable_cpu_diff());
			}
			DBGPRT(LOG(INFO)<<"444");
			this->net_->Update();
			DBGPRT(LOG(INFO)<<"555");
			for(int i=0;i<this->childProcessSum;++i){
				if(this->flagComputeEndNeedUpdate[i]==1){
					for(int param_id = 0; param_id < net_params.size(); ++param_id){
						caffe_copy(net_params[param_id]->count(),
								net_params[param_id]->cpu_data(),this->tempdata[i][param_id]->mutable_cpu_data());
					}
				}
			}
			DBGPRT(LOG(INFO)<<"666");
		}
	template <typename Dtype>
		void SGDSolver<Dtype>::ComputeUpdateValueServerThreadGPU(){
			vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
			vector<float>& net_params_lr = this->net_->params_lr();
			vector<float>& net_params_weight_decay = this->net_->params_weight_decay();
			Dtype rate = GetLearningRate();
			if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
				LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
			}
			Dtype momentum = this->param_.momentum();
			Dtype weight_decay = this->param_.weight_decay();
			string regularization_type = this->param_.regularization_type();
			DBGPRT(LOG(INFO)<<"111");
			for(int param_id = 0; param_id < net_params.size(); ++param_id){
				cudaMemset(net_params[param_id]->mutable_gpu_diff(),0,sizeof(Dtype)*(net_params[param_id]->count()));
			}
			DBGPRT(LOG(INFO)<<"222");
			for(int i=0;i<this->childProcessSum;++i){
				if(this->flagComputeEndNeedUpdate[i]==1){
					for(int param_id = 0; param_id < net_params.size(); ++param_id){
						caffe_gpu_axpy(net_params[param_id]->count(),(Dtype)1,
								this->tempdata[i][param_id]->mutable_gpu_data(),net_params[param_id]->mutable_gpu_diff());
					}
				}
			}
			DBGPRT(LOG(INFO)<<"333");
			assert(currentUpdateCount==0);
			for (int param_id = 0; param_id < net_params.size(); ++param_id) {
				caffe_gpu_scal(net_params[param_id]->count(),(Dtype)(1.0f/currentUpdateCount),net_params[param_id]->mutable_gpu_diff());
				Dtype local_rate = rate * net_params_lr[param_id];
				Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
				if (local_decay) {
					if (regularization_type == "L2") {
						caffe_gpu_axpy(net_params[param_id]->count(),
								local_decay,
								net_params[param_id]->gpu_data(),
								net_params[param_id]->mutable_gpu_diff());
					} else if (regularization_type == "L1") {
						caffe_gpu_sign(net_params[param_id]->count(),
								net_params[param_id]->gpu_data(),
								temp_[param_id]->mutable_gpu_data());
						caffe_gpu_axpy(net_params[param_id]->count(),
								local_decay,
								temp_[param_id]->gpu_data(),
								net_params[param_id]->mutable_gpu_diff());
					} else {
						LOG(FATAL) << "Unknown regularization type: " << regularization_type;
					}
				}

				caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
						net_params[param_id]->gpu_diff(), momentum,
						history_[param_id]->mutable_gpu_data());
				caffe_copy(net_params[param_id]->count(),
						history_[param_id]->gpu_data(),
						net_params[param_id]->mutable_gpu_diff());
			}
			DBGPRT(LOG(INFO)<<"444");
			this->net_->Update();
			DBGPRT(LOG(INFO)<<"555");
			for(int i=0;i<this->childProcessSum;++i){
				if(this->flagComputeEndNeedUpdate[i]==1){
#ifdef DIRECTGPU
					for(int param_id = 0; param_id < net_params.size(); ++param_id){
						caffe_copy(net_params[param_id]->count(),
								net_params[param_id]->gpu_data(),
								this->tempdata[i][param_id]->mutable_gpu_data());
					}
#else
					for(int param_id = 0; param_id < net_params.size(); ++param_id){
						caffe_copy(net_params[param_id]->count(),
								net_params[param_id]->cpu_data(),
								this->tempdata[i][param_id]->mutable_cpu_data());
					}
#endif
				}
			}
			DBGPRT(LOG(INFO)<<"666");
		}
	template <typename Dtype>
		void ComputeValueThreadClient(Solver<Dtype>* layer, const int tid, const int endNum) {
			int startNum = 0;
			CHECK(layer);
			for(int i=startNum; i < endNum; ++i){
				layer->ComputeValueClient(tid);
			}
			LOG(INFO)<<"Thread fin "<<tid;
		}
	template <typename Dtype>
		void Solver<Dtype>::ComputeValueClient(int tid){
			int mpi_source;
			ComputeUpdateValueClientThread(mpi_source,tid);
			{
				unique_lock<mutex> ul(mutexUpdateNet);
				condUpdateNet.wait(ul, []{return currentUpdateCount == 0;});
			}
			flagComputeEndNeedUpdate[tid-1]=0;
			vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
			for (int param_id = 0; param_id < net_params.size(); ++param_id) {
#ifdef DIRECTGPU
				caffe_mpi_send<Dtype>(tempdata[tid-1][param_id]->mutable_gpu_data(),tempdata[tid-1][param_id]->count(),
						tid,TAG_NET_OUT,MPI_COMM_WORLD);
#else
				caffe_mpi_send<Dtype>(tempdata[tid-1][param_id]->mutable_cpu_data(),tempdata[tid-1][param_id]->count(),
						tid,TAG_NET_OUT,MPI_COMM_WORLD);
#endif
			}	
		}
	template <typename Dtype>
		void SGDSolver<Dtype>::ComputeUpdateValueClientThread(int& mpi_source,int tid){
			GetValue(mpi_source,tid);
		}
	template <typename Dtype>
		Solver<Dtype>::Solver(const SolverParameter& param)
		: net_() {
			Init(param);
		}

	template <typename Dtype>
		Solver<Dtype>::Solver(const string& param_file)
		: net_() {
			SolverParameter param;
			ReadProtoFromTextFileOrDie(param_file, &param);
			Init(param);
		}

	template <typename Dtype>
		SGDSolver<Dtype>::~SGDSolver(){
			if(this->rank==0){
				delete [] this->flagComputeEndNeedUpdate;
				vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
				for(int i=0;i<this->childProcessSum;++i){
					for(int j=0;j<net_params.size();++j)
						delete this->tempdata[i][j];
					delete[] this->tempdata[i];
				}
				delete [] this->tempdata;
			}else{
			}
		}
	template <typename Dtype>
		void Solver<Dtype>::Init(const SolverParameter& param) {
			LOG(INFO) << "Initializing solver from parameters: " << std::endl
				<< param.DebugString();
			param_ = param;
			if (param_.random_seed() >= 0) {
				Caffe::set_random_seed(param_.random_seed());
			}
			// Scaffolding code
			InitTrainNet();
			InitTestNets();
			int size;
			MPI_Comm_rank (MPI_COMM_WORLD, &rank);
			MPI_Comm_size (MPI_COMM_WORLD, &size);
			LOG(INFO) << "Solver scaffolding done.";
		}

	template <typename Dtype>
		void Solver<Dtype>::InitTrainNet() {
			const int num_train_nets = param_.has_net() + param_.has_net_param() +
				param_.has_train_net() + param_.has_train_net_param();
			const string& field_names = "net, net_param, train_net, train_net_param";
			CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
				<< "using one of these fields: " << field_names;
			CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
				<< "one of these fields specifying a train_net: " << field_names;
			NetParameter net_param;
			if (param_.has_train_net_param()) {
				LOG(INFO) << "Creating training net specified in train_net_param.";
				net_param.CopyFrom(param_.train_net_param());
			} else if (param_.has_train_net()) {
				LOG(INFO) << "Creating training net from train_net file: "
					<< param_.train_net();
				ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
			}
			if (param_.has_net_param()) {
				LOG(INFO) << "Creating training net specified in net_param.";
				net_param.CopyFrom(param_.net_param());
			}
			if (param_.has_net()) {
				LOG(INFO) << "Creating training net from net file: " << param_.net();
				ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
			}
			// Set the correct NetState.  We start with the solver defaults (lowest
			// precedence); then, merge in any NetState specified by the net_param itself;
			// finally, merge in any NetState specified by the train_state (highest
			// precedence).
			NetState net_state;
			net_state.set_phase(TRAIN);
			net_state.MergeFrom(net_param.state());
			net_state.MergeFrom(param_.train_state());
			net_param.mutable_state()->CopyFrom(net_state);
			net_.reset(new Net<Dtype>(net_param));
		}

	template <typename Dtype>
		void Solver<Dtype>::InitTestNets() {
			const bool has_net_param = param_.has_net_param();
			const bool has_net_file = param_.has_net();
			const int num_generic_nets = has_net_param + has_net_file;
			CHECK_LE(num_generic_nets, 1)
				<< "Both net_param and net_file may not be specified.";
			const int num_test_net_params = param_.test_net_param_size();
			const int num_test_net_files = param_.test_net_size();
			const int num_test_nets = num_test_net_params + num_test_net_files;
			if (num_generic_nets) {
				CHECK_GE(param_.test_iter_size(), num_test_nets)
					<< "test_iter must be specified for each test network.";
			} else {
				CHECK_EQ(param_.test_iter_size(), num_test_nets)
					<< "test_iter must be specified for each test network.";
			}
			// If we have a generic net (specified by net or net_param, rather than
			// test_net or test_net_param), we may have an unlimited number of actual
			// test networks -- the actual number is given by the number of remaining
			// test_iters after any test nets specified by test_net_param and/or test_net
			// are evaluated.
			const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;
			const int num_test_net_instances = num_test_nets + num_generic_net_instances;
			if (param_.test_state_size()) {
				CHECK_EQ(param_.test_state_size(), num_test_net_instances)
					<< "test_state must be unspecified or specified once per test net.";
			}
			if (num_test_net_instances) {
				CHECK_GT(param_.test_interval(), 0);
			}
			int test_net_id = 0;
			vector<string> sources(num_test_net_instances);
			vector<NetParameter> net_params(num_test_net_instances);
			for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {
				sources[test_net_id] = "test_net_param";
				net_params[test_net_id].CopyFrom(param_.test_net_param(i));
			}
			for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {
				sources[test_net_id] = "test_net file: " + param_.test_net(i);
				ReadNetParamsFromTextFileOrDie(param_.test_net(i),
						&net_params[test_net_id]);
			}
			const int remaining_test_nets = param_.test_iter_size() - test_net_id;
			if (has_net_param) {
				for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
					sources[test_net_id] = "net_param";
					net_params[test_net_id].CopyFrom(param_.net_param());
				}
			}
			if (has_net_file) {
				for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
					sources[test_net_id] = "net file: " + param_.net();
					ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
				}
			}
			test_nets_.resize(num_test_net_instances);
			for (int i = 0; i < num_test_net_instances; ++i) {
				// Set the correct NetState.  We start with the solver defaults (lowest
				// precedence); then, merge in any NetState specified by the net_param
				// itself; finally, merge in any NetState specified by the test_state
				// (highest precedence).
				NetState net_state;
				net_state.set_phase(TEST);
				net_state.MergeFrom(net_params[i].state());
				if (param_.test_state_size()) {
					net_state.MergeFrom(param_.test_state(i));
				}
				net_params[i].mutable_state()->CopyFrom(net_state);
				LOG(INFO)
					<< "Creating test net (#" << i << ") specified by " << sources[i];
				test_nets_[i].reset(new Net<Dtype>(net_params[i]));
			}
		}

	template <typename Dtype>
		void Solver<Dtype>::Solve(const char* resume_file) {
			Caffe::set_phase(Caffe::TRAIN);
			LOG(INFO) << "Solving " << net_->name();
			PreSolve();

			iter_ = 0;
			if (resume_file) {
				LOG(INFO) << "Restoring previous solver status from " << resume_file;
				Restore(resume_file);
			}
			// Remember the initial iter_ value; will be non-zero if we loaded from a
			// resume_file above.
			const int start_iter = iter_;

			// For a network that is trained by the solver, no bottom or top vecs
			// should be given, and we will just provide dummy vecs.
			vector<Blob<Dtype>*> bottom_vec;
			vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
			int msize;
			MPI_Comm_size (MPI_COMM_WORLD, &msize);
			undoneIter = param_.max_iter() - iter_;
			childProcessSum = msize -1 ;
			if(rank==0){
				flagComputeEndNeedUpdate=new int[childProcessSum]();
				tempdata= new Bloblite<Dtype> **[childProcessSum];
				for(int i=0;i<childProcessSum;++i){
					tempdata[i]=new Bloblite<Dtype>* [net_params.size()];
					for(int j=0;j<net_params.size();++j){
						tempdata[i][j]= new Bloblite<Dtype>(net_params[j]->count());
					}
				}

				MPI_Barrier(MPI_COMM_WORLD);
				int itSize;
				itSize = param_.max_iter() / childProcessSum;
				int tailSize;
				tailSize = param_.max_iter() % childProcessSum;
				thread threadServer(&ComputeValueThreadServer<Dtype>,this,childProcessSum);
				//thread * threadClientUpdate = new thread[childProcessSum];
				vector<thread> threadClientUpdate(childProcessSum);
				for(int i=0;i< childProcessSum;++i){
					threadClientUpdate[i] = std::thread(&ComputeValueThreadClient<Dtype>,this,i+1,(i<tailSize ? itSize +1:itSize));
				}
				//thread * threadRead = new thread[childProcessSum];
				vector<thread> threadRead(childProcessSum);
				for(int i=0;i< childProcessSum;++i){
					threadRead[i] = std::thread(&ReadDataFromDBandDistribute<Dtype>,net_->layers().at(0).get(),
							i+1,(i<tailSize ? itSize +1:itSize),childProcessSum);
				}
				for(int i=0;i<childProcessSum;++i){
					threadRead[i].join();
					threadClientUpdate[i].join();
				}
				threadServer.join();
				threadClientUpdate.clear();
				threadRead.clear();
				//delete[] threadClientUpdate;
				//delete[] threadRead;
			}else{
				int iterSize;
				iterSize = param_.max_iter() / childProcessSum;
				int tailtSize;
				tailtSize = param_.max_iter() % childProcessSum;
				int startNum,endNum;
				startNum = 0;
				endNum = iterSize;
				if(tailtSize >= rank) ++endNum;
				MPI_Barrier(MPI_COMM_WORLD);
#ifdef ASYNCTRAN
				semaphore semRead;
				semaphore semNext;
				thread threadRead(SlaveReadData<Dtype>,net_,endNum,&semRead,&semNext);//////
#endif
				for(int i=startNum; i< endNum; ++i){
					MPI_Status status;
					status.MPI_ERROR=0;
					iter_ = (rank - 1) + i * childProcessSum;
					net_->taskiter = iter_;
#ifdef ASYNCTRAN
					semRead.wait();
					DBGPRT(LOG(INFO)<<"FB START "<<i<<" "<<iter_);
					Dtype loss = net_->ForwardBackward(bottom_vec,&semNext);
					DBGPRT(LOG(INFO)<<"FB FIN CUVC START "<<i<<" "<<iter_);
					semNext.notify();//TODO I think this statement should at net->Forward when 2nd net forward is finished,but the accuracy is error
#else
					DBGPRT(LOG(INFO)<<"FB START "<<i<<" "<<iter_);
					Dtype loss = net_->ForwardBackward(bottom_vec);
					DBGPRT(LOG(INFO)<<"FB FIN CUVC START "<<i<<" "<<iter_);
#endif
					ComputeUpdateValueClient();
					DBGPRT(LOG(INFO)<<"CUVC FIN RECV NET START "<<i<<" "<<iter_);
					//			memset(&status,0,sizeof(status));
					vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
#ifdef DIRECTGPU
					for (int param_id = 0; param_id < net_params.size(); ++param_id) {
						caffe_mpi_recv<Dtype>(net_params[param_id]->mutable_gpu_data(),net_params[param_id]->count(),
								0,TAG_NET_OUT,MPI_COMM_WORLD,&status);
					}
#else
					for (int param_id = 0; param_id < net_params.size(); ++param_id) {
						caffe_mpi_recv<Dtype>(net_params[param_id]->mutable_cpu_data(),net_params[param_id]->count(),
								0,TAG_NET_OUT,MPI_COMM_WORLD,&status);
					}
#endif
					DBGPRT(LOG(INFO)<<"RECV NET FIN "<<i<<" "<<iter_);
					// Save a snapshot if needed.
					if (param_.snapshot() && iter_ > start_iter &&
							iter_ % param_.snapshot() == 0) {
						Snapshot();
					}

					const bool display = param_.display() && iter_ % param_.display() == 0;
					net_->set_debug_info(display && param_.debug_info());
					if (display) {
						LOG(INFO) << "Iteration " << iter_ << ", loss = " << loss;
						const vector<Blob<Dtype>*>& result = net_->output_blobs();
						int score_index = 0;
						for (int j = 0; j < result.size(); ++j) {
							const Dtype* result_vec = result[j]->cpu_data();
							const string& output_name =
								net_->blob_names()[net_->output_blob_indices()[j]];
							const Dtype loss_weight =
								net_->blob_loss_weights()[net_->output_blob_indices()[j]];
							for (int k = 0; k < result[j]->count(); ++k) {
								ostringstream loss_msg_stream;
								if (loss_weight) {
									loss_msg_stream << " (* " << loss_weight
										<< " = " << loss_weight * result_vec[k] << " loss)";
								}
								LOG(INFO) << "    Train net output #"
									<< score_index++ << ": " << output_name << " = "
									<< result_vec[k] << loss_msg_stream.str();
							}
						}
					}
				}
#ifdef ASYNCTRAN
				threadRead.join();//////
#endif
			}
			MPI_Barrier(MPI_COMM_WORLD);
			if(rank==0){
				// Always save a snapshot after optimization, unless overridden by setting
				// snapshot_after_train := false.
				if (param_.snapshot_after_train()) { Snapshot(); }
				// After the optimization is done, run an additional train and test pass to
				// display the train and test loss/outputs if appropriate (based on the
				// display and test_interval settings, respectively).  Unlike in the rest of
				// training, for the train net we only run a forward pass as we've already
				// updated the parameters "max_iter" times -- this final pass is only done to
				// display the loss, which is computed in the forward pass.
				if (param_.display() && iter_ % param_.display() == 0) {
					Dtype loss;
					net_->taskiter=0;
					net_->ForwardTest(bottom_vec, &loss);
					LOG(INFO) << "Iteration " << iter_ << ", loss = " << loss;
				}
				if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
					TestAll();
				}
			}
		}

	template <typename Dtype>
		void Solver<Dtype>::TestAll() {
			for (int test_net_id = 0; test_net_id < test_nets_.size(); ++test_net_id) {
				Test(test_net_id);
			}
		}


	template <typename Dtype>
		void Solver<Dtype>::Test(const int test_net_id) {
			LOG(INFO) << "Iteration " << iter_
				<< ", Testing net (#" << test_net_id << ")";
			// We need to set phase to test before running.
			Caffe::set_phase(Caffe::TEST);
			CHECK_NOTNULL(test_nets_[test_net_id].get())->
				ShareTrainedLayersWith(net_.get());
			vector<Dtype> test_score;
			vector<int> test_score_output_id;
			vector<Blob<Dtype>*> bottom_vec;
			const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
			Dtype loss = 0;
			for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
				Dtype iter_loss;
				const vector<Blob<Dtype>*>& result =
					test_net->ForwardTest(bottom_vec, &iter_loss);
				if (param_.test_compute_loss()) {
					loss += iter_loss;
				}
				if (i == 0) {
					for (int j = 0; j < result.size(); ++j) {
						const Dtype* result_vec = result[j]->cpu_data();
						for (int k = 0; k < result[j]->count(); ++k) {
							test_score.push_back(result_vec[k]);
							test_score_output_id.push_back(j);
						}
					}
				} else {
					int idx = 0;
					for (int j = 0; j < result.size(); ++j) {
						const Dtype* result_vec = result[j]->cpu_data();
						for (int k = 0; k < result[j]->count(); ++k) {
							test_score[idx++] += result_vec[k];
						}
					}
				}
			}
			if (param_.test_compute_loss()) {
				loss /= param_.test_iter(test_net_id);
				LOG(INFO) << "Test loss: " << loss;
			}
			for (int i = 0; i < test_score.size(); ++i) {
				const int output_blob_index =
					test_net->output_blob_indices()[test_score_output_id[i]];
				const string& output_name = test_net->blob_names()[output_blob_index];
				const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
				ostringstream loss_msg_stream;
				const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
				if (loss_weight) {
					loss_msg_stream << " (* " << loss_weight
						<< " = " << loss_weight * mean_score << " loss)";
				}
				LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
					<< mean_score << loss_msg_stream.str();
			}
			Caffe::set_phase(Caffe::TRAIN);
		}


	template <typename Dtype>
		void Solver<Dtype>::Snapshot() {
			NetParameter net_param;
			// For intermediate results, we will also dump the gradient values.
			net_->ToProto(&net_param, param_.snapshot_diff());
			string filename(param_.snapshot_prefix());
			string model_filename, snapshot_filename;
			const int kBufferSize = 20;
			char iter_str_buffer[kBufferSize];
			snprintf(iter_str_buffer, kBufferSize, "_iter_%d", iter_);
			filename += iter_str_buffer;
			model_filename = filename + ".caffemodel";
			LOG(INFO) << "Snapshotting to " << model_filename;
			WriteProtoToBinaryFile(net_param, model_filename.c_str());
			SolverState state;
			SnapshotSolverState(&state);
			state.set_iter(iter_);
			state.set_learned_net(model_filename);
			snapshot_filename = filename + ".solverstate";
			LOG(INFO) << "Snapshotting solver state to " << snapshot_filename;
			WriteProtoToBinaryFile(state, snapshot_filename.c_str());
		}

	template <typename Dtype>
		void Solver<Dtype>::Restore(const char* state_file) {
			SolverState state;
			NetParameter net_param;
			ReadProtoFromBinaryFile(state_file, &state);
			if (state.has_learned_net()) {
				ReadProtoFromBinaryFile(state.learned_net().c_str(), &net_param);
				net_->CopyTrainedLayersFrom(net_param);
			}
			iter_ = state.iter();
			RestoreSolverState(state);
		}


	// Return the current learning rate. The currently implemented learning rate
	// policies are as follows:
	//    - fixed: always return base_lr.
	//    - step: return base_lr * gamma ^ (floor(iter / step))
	//    - exp: return base_lr * gamma ^ iter
	//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
	// where base_lr, gamma, step and power are defined in the solver parameter
	// protocol buffer, and iter is the current iteration.
	template <typename Dtype>
		Dtype SGDSolver<Dtype>::GetLearningRate() {
			Dtype rate;
			const string& lr_policy = this->param_.lr_policy();
			if (lr_policy == "fixed") {
				rate = this->param_.base_lr();
			} else if (lr_policy == "step") {
				int current_step = this->iter_ / this->param_.stepsize();
				rate = this->param_.base_lr() *
					pow(this->param_.gamma(), current_step);
			} else if (lr_policy == "exp") {
				rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
			} else if (lr_policy == "inv") {
				rate = this->param_.base_lr() *
					pow(Dtype(1) + this->param_.gamma() * this->iter_,
							- this->param_.power());
			} else {
				LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
			}
			return rate;
		}


	template <typename Dtype>
		void SGDSolver<Dtype>::PreSolve() {
			// Initialize the history
			vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
			history_.clear();
			update_.clear();
			temp_.clear();
			for (int i = 0; i < net_params.size(); ++i) {
				const Blob<Dtype>* net_param = net_params[i].get();
				history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(
								net_param->num(), net_param->channels(), net_param->height(),
								net_param->width())));
				update_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(
								net_param->num(), net_param->channels(), net_param->height(),
								net_param->width())));
				temp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(
								net_param->num(), net_param->channels(), net_param->height(),
								net_param->width())));
			}
		}


	template <typename Dtype>
		void SGDSolver<Dtype>::ComputeUpdateValueClient() {
			Dtype rate = GetLearningRate();
			if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
				LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
			}
		}
#if 0
	template <typename Dtype>
		void SGDSolver<Dtype>::ComputeUpdateValue() {
			vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
			vector<float>& net_params_lr = this->net_->params_lr();
			vector<float>& net_params_weight_decay = this->net_->params_weight_decay();
			// get the learning rate
			Dtype rate = GetLearningRate();
			if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
				LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
			}
			Dtype momentum = this->param_.momentum();
			Dtype weight_decay = this->param_.weight_decay();
			string regularization_type = this->param_.regularization_type();
			switch (Caffe::mode()) {
				case Caffe::CPU:
					for (int param_id = 0; param_id < net_params.size(); ++param_id) {
						// Compute the value to history, and then copy them to the blob's diff.
						Dtype local_rate = rate * net_params_lr[param_id];
						Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

						if (local_decay) {
							if (regularization_type == "L2") {
								// add weight decay
								caffe_axpy(net_params[param_id]->count(),
										local_decay,
										net_params[param_id]->cpu_data(),
										net_params[param_id]->mutable_cpu_diff());
							} else if (regularization_type == "L1") {
								caffe_cpu_sign(net_params[param_id]->count(),
										net_params[param_id]->cpu_data(),
										temp_[param_id]->mutable_cpu_data());
								caffe_axpy(net_params[param_id]->count(),
										local_decay,
										temp_[param_id]->cpu_data(),
										net_params[param_id]->mutable_cpu_diff());
							} else {
								LOG(FATAL) << "Unknown regularization type: " << regularization_type;
							}
						}

						caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
								net_params[param_id]->cpu_diff(), momentum,
								history_[param_id]->mutable_cpu_data());
						// copy
						caffe_copy(net_params[param_id]->count(),
								history_[param_id]->cpu_data(),
								net_params[param_id]->mutable_cpu_diff());
					}
					break;
				case Caffe::GPU:
#ifndef CPU_ONLY
					for (int param_id = 0; param_id < net_params.size(); ++param_id) {
						// Compute the value to history, and then copy them to the blob's diff.
						Dtype local_rate = rate * net_params_lr[param_id];
						Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

						if (local_decay) {
							if (regularization_type == "L2") {
								// add weight decay
								caffe_gpu_axpy(net_params[param_id]->count(),
										local_decay,
										net_params[param_id]->gpu_data(),
										net_params[param_id]->mutable_gpu_diff());
							} else if (regularization_type == "L1") {
								caffe_gpu_sign(net_params[param_id]->count(),
										net_params[param_id]->gpu_data(),
										temp_[param_id]->mutable_gpu_data());
								caffe_gpu_axpy(net_params[param_id]->count(),
										local_decay,
										temp_[param_id]->gpu_data(),
										net_params[param_id]->mutable_gpu_diff());
							} else {
								LOG(FATAL) << "Unknown regularization type: " << regularization_type;
							}
						}

						caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
								net_params[param_id]->gpu_diff(), momentum,
								history_[param_id]->mutable_gpu_data());
						// copy
						caffe_copy(net_params[param_id]->count(),
								history_[param_id]->gpu_data(),
								net_params[param_id]->mutable_gpu_diff());
					}
#else
					NO_GPU;
#endif
					break;
				default:
					LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
			}
		}
#endif
	template <typename Dtype>
		void SGDSolver<Dtype>::SnapshotSolverState(SolverState* state) {
			state->clear_history();
			for (int i = 0; i < history_.size(); ++i) {
				// Add history
				BlobProto* history_blob = state->add_history();
				history_[i]->ToProto(history_blob);
			}
		}

	template <typename Dtype>
		void SGDSolver<Dtype>::RestoreSolverState(const SolverState& state) {
			CHECK_EQ(state.history_size(), history_.size())
				<< "Incorrect length of history blobs.";
			LOG(INFO) << "SGDSolver: restoring history";
			for (int i = 0; i < history_.size(); ++i) {
				history_[i]->FromProto(state.history(i));
			}
		}

	template <typename Dtype>
		void NesterovSolver<Dtype>::ComputeUpdateValue() {
			vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
			vector<float>& net_params_lr = this->net_->params_lr();
			vector<float>& net_params_weight_decay = this->net_->params_weight_decay();
			// get the learning rate
			Dtype rate = this->GetLearningRate();
			if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
				LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
			}
			Dtype momentum = this->param_.momentum();
			Dtype weight_decay = this->param_.weight_decay();
			string regularization_type = this->param_.regularization_type();
			switch (Caffe::mode()) {
				case Caffe::CPU:
					for (int param_id = 0; param_id < net_params.size(); ++param_id) {
						// save history momentum for stepping back
						caffe_copy(net_params[param_id]->count(),
								this->history_[param_id]->cpu_data(),
								this->update_[param_id]->mutable_cpu_data());

						Dtype local_rate = rate * net_params_lr[param_id];
						Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

						if (local_decay) {
							if (regularization_type == "L2") {
								// add weight decay
								caffe_axpy(net_params[param_id]->count(),
										local_decay,
										net_params[param_id]->cpu_data(),
										net_params[param_id]->mutable_cpu_diff());
							} else if (regularization_type == "L1") {
								caffe_cpu_sign(net_params[param_id]->count(),
										net_params[param_id]->cpu_data(),
										this->temp_[param_id]->mutable_cpu_data());
								caffe_axpy(net_params[param_id]->count(),
										local_decay,
										this->temp_[param_id]->cpu_data(),
										net_params[param_id]->mutable_cpu_diff());
							} else {
								LOG(FATAL) << "Unknown regularization type: " << regularization_type;
							}
						}

						// update history
						caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
								net_params[param_id]->cpu_diff(), momentum,
								this->history_[param_id]->mutable_cpu_data());

						// compute udpate: step back then over step
						caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1) + momentum,
								this->history_[param_id]->cpu_data(), -momentum,
								this->update_[param_id]->mutable_cpu_data());

						// copy
						caffe_copy(net_params[param_id]->count(),
								this->update_[param_id]->cpu_data(),
								net_params[param_id]->mutable_cpu_diff());
					}
					break;
				case Caffe::GPU:
#ifndef CPU_ONLY
					for (int param_id = 0; param_id < net_params.size(); ++param_id) {
						// save history momentum for stepping back
						caffe_copy(net_params[param_id]->count(),
								this->history_[param_id]->gpu_data(),
								this->update_[param_id]->mutable_gpu_data());

						Dtype local_rate = rate * net_params_lr[param_id];
						Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

						if (local_decay) {
							if (regularization_type == "L2") {
								// add weight decay
								caffe_gpu_axpy(net_params[param_id]->count(),
										local_decay,
										net_params[param_id]->gpu_data(),
										net_params[param_id]->mutable_gpu_diff());
							} else if (regularization_type == "L1") {
								caffe_gpu_sign(net_params[param_id]->count(),
										net_params[param_id]->gpu_data(),
										this->temp_[param_id]->mutable_gpu_data());
								caffe_gpu_axpy(net_params[param_id]->count(),
										local_decay,
										this->temp_[param_id]->gpu_data(),
										net_params[param_id]->mutable_gpu_diff());
							} else {
								LOG(FATAL) << "Unknown regularization type: " << regularization_type;
							}
						}

						// update history
						caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
								net_params[param_id]->gpu_diff(), momentum,
								this->history_[param_id]->mutable_gpu_data());

						// compute udpate: step back then over step
						caffe_gpu_axpby(net_params[param_id]->count(), Dtype(1) + momentum,
								this->history_[param_id]->gpu_data(), -momentum,
								this->update_[param_id]->mutable_gpu_data());

						// copy
						caffe_copy(net_params[param_id]->count(),
								this->update_[param_id]->gpu_data(),
								net_params[param_id]->mutable_gpu_diff());
					}
#else
					NO_GPU;
#endif
					break;
				default:
					LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
			}
		}

	template <typename Dtype>
		void AdaGradSolver<Dtype>::ComputeUpdateValue() {
			vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
			vector<float>& net_params_lr = this->net_->params_lr();
			vector<float>& net_params_weight_decay = this->net_->params_weight_decay();
			// get the learning rate
			Dtype rate = this->GetLearningRate();
			Dtype delta = this->param_.delta();
			if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
				LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
			}
			Dtype weight_decay = this->param_.weight_decay();
			string regularization_type = this->param_.regularization_type();
			switch (Caffe::mode()) {
				case Caffe::CPU:
					for (int param_id = 0; param_id < net_params.size(); ++param_id) {
						Dtype local_rate = rate * net_params_lr[param_id];
						Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

						if (local_decay) {
							if (regularization_type == "L2") {
								// add weight decay
								caffe_axpy(net_params[param_id]->count(),
										local_decay,
										net_params[param_id]->cpu_data(),
										net_params[param_id]->mutable_cpu_diff());
							} else if (regularization_type == "L1") {
								caffe_cpu_sign(net_params[param_id]->count(),
										net_params[param_id]->cpu_data(),
										this->temp_[param_id]->mutable_cpu_data());
								caffe_axpy(net_params[param_id]->count(),
										local_decay,
										this->temp_[param_id]->cpu_data(),
										net_params[param_id]->mutable_cpu_diff());
							} else {
								LOG(FATAL) << "Unknown regularization type: " << regularization_type;
							}
						}

						// compute square of gradient in update
						caffe_powx(net_params[param_id]->count(),
								net_params[param_id]->cpu_diff(), Dtype(2),
								this->update_[param_id]->mutable_cpu_data());

						// update history
						caffe_add(net_params[param_id]->count(),
								this->update_[param_id]->cpu_data(),
								this->history_[param_id]->cpu_data(),
								this->history_[param_id]->mutable_cpu_data());

						// prepare update
						caffe_powx(net_params[param_id]->count(),
								this->history_[param_id]->cpu_data(), Dtype(0.5),
								this->update_[param_id]->mutable_cpu_data());

						caffe_add_scalar(net_params[param_id]->count(),
								delta, this->update_[param_id]->mutable_cpu_data());

						caffe_div(net_params[param_id]->count(),
								net_params[param_id]->cpu_diff(),
								this->update_[param_id]->cpu_data(),
								this->update_[param_id]->mutable_cpu_data());

						// scale and copy
						caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
								this->update_[param_id]->cpu_data(), Dtype(0),
								net_params[param_id]->mutable_cpu_diff());
					}
					break;
				case Caffe::GPU:
#ifndef CPU_ONLY
					for (int param_id = 0; param_id < net_params.size(); ++param_id) {
						Dtype local_rate = rate * net_params_lr[param_id];
						Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

						if (local_decay) {
							if (regularization_type == "L2") {
								// add weight decay
								caffe_gpu_axpy(net_params[param_id]->count(),
										local_decay,
										net_params[param_id]->gpu_data(),
										net_params[param_id]->mutable_gpu_diff());
							} else if (regularization_type == "L1") {
								caffe_gpu_sign(net_params[param_id]->count(),
										net_params[param_id]->gpu_data(),
										this->temp_[param_id]->mutable_gpu_data());
								caffe_gpu_axpy(net_params[param_id]->count(),
										local_decay,
										this->temp_[param_id]->gpu_data(),
										net_params[param_id]->mutable_gpu_diff());
							} else {
								LOG(FATAL) << "Unknown regularization type: " << regularization_type;
							}
						}

						// compute square of gradient in update
						caffe_gpu_powx(net_params[param_id]->count(),
								net_params[param_id]->gpu_diff(), Dtype(2),
								this->update_[param_id]->mutable_gpu_data());

						// update history
						caffe_gpu_add(net_params[param_id]->count(),
								this->update_[param_id]->gpu_data(),
								this->history_[param_id]->gpu_data(),
								this->history_[param_id]->mutable_gpu_data());

						// prepare update
						caffe_gpu_powx(net_params[param_id]->count(),
								this->history_[param_id]->gpu_data(), Dtype(0.5),
								this->update_[param_id]->mutable_gpu_data());

						caffe_gpu_add_scalar(net_params[param_id]->count(),
								delta, this->update_[param_id]->mutable_gpu_data());

						caffe_gpu_div(net_params[param_id]->count(),
								net_params[param_id]->gpu_diff(),
								this->update_[param_id]->gpu_data(),
								this->update_[param_id]->mutable_gpu_data());

						// scale and copy
						caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
								this->update_[param_id]->gpu_data(), Dtype(0),
								net_params[param_id]->mutable_gpu_diff());
					}
#else
					NO_GPU;
#endif
					break;
				default:
					LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
			}
		}
	template <typename Dtype>
		void SGDSolver<Dtype>::GetValue(int &mpi_source,const int tid) {
			MPI_Status status;

			vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
			for (int param_id = net_params.size()-1; param_id >= 0; --param_id) {
				memset(&status,0,sizeof(status));
//assert(net_params[param_id]->count()!=this->tempdata[tid-1][param_id]->count());
#ifdef DIRECTGPU
				if(param_id==net_params.size()-1){
					caffe_mpi_recv<Dtype>(this->tempdata[tid-1][param_id]->mutable_gpu_data(),net_params[param_id]->count(),
							tid,TAG_UPDATE_1,MPI_COMM_WORLD,&status);
				}else{
					caffe_mpi_recv<Dtype>(this->tempdata[tid-1][param_id]->mutable_gpu_data(),net_params[param_id]->count(),
							tid,TAG_UPDATE,MPI_COMM_WORLD,&status);
				}
#else
				if(param_id==net_params.size()-1){
					caffe_mpi_recv<Dtype>(this->tempdata[tid-1][param_id]->mutable_cpu_data(),net_params[param_id]->count(),
							tid,TAG_UPDATE_1,MPI_COMM_WORLD,&status);
				}else{
					caffe_mpi_recv<Dtype>(this->tempdata[tid-1][param_id]->mutable_cpu_data(),net_params[param_id]->count(),
							tid,TAG_UPDATE,MPI_COMM_WORLD,&status);
				}
#endif
			}

			{
				unique_lock<mutex> lk(mutexUpdateWeight);
				this->flagComputeEndNeedUpdate[tid-1] = 1;
				++currentUpdateCount;
				condUpdateWeight.notify_all();
			}
		}

	INSTANTIATE_CLASS(Solver);
	INSTANTIATE_CLASS(SGDSolver);
	INSTANTIATE_CLASS(NesterovSolver);
	INSTANTIATE_CLASS(AdaGradSolver);

}  // namespace caffe
