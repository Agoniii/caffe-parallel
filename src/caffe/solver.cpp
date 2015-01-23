#include <cstdio>

#include <algorithm>
#include <string>
#include <vector>
#include <queue>
#include <sys/time.h>

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

std::queue<int> idleQ;
sem_t semQ;//wait program finish
int taskS1;
int upNum=0;
int upSum;
//void * tempData=NULL;//like float/double  //test del tempData20150113
void * tempDiff=NULL;//like float/double
int *flagCC=NULL;
pthread_mutex_t mutexFin;//=PTHREAD_MUTEX_INITIALIZER;//check and wait program finish
pthread_cond_t condFin;//=PTHREAD_MUTEX_INITIALIZER;//check and wait program finish
pthread_mutex_t mutexUp=PTHREAD_MUTEX_INITIALIZER;//wait update net paramater in server thread
pthread_cond_t condUp=PTHREAD_COND_INITIALIZER;//wait update net paramater in server thread
pthread_mutex_t mutexCtrl=PTHREAD_MUTEX_INITIALIZER;//when update net paramaters finished, broadcast to send data to MPI clients
pthread_cond_t condCtrl=PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutexData=PTHREAD_MUTEX_INITIALIZER;//update data and diff
atomInt taskSum,taskS;
int itest=0;
namespace caffe {
template <typename Dtype>
	void* ComputeValueThreadServer(void* param) {
		SGDSolver<Dtype>* layer = static_cast<SGDSolver<Dtype>*>( ((tprama*) param)->layer);
		//int tid = ((tprama*)param)->tid;
		struct timeval now_time;
		struct timespec wait_time;
		int timeoutret;
		while(true){
			if(taskSum.getValue() <=0){LOG(INFO)<<"Server out"; pthread_exit(NULL); }
			gettimeofday(&now_time,NULL);
			wait_time.tv_sec      =       now_time.tv_sec + WAIT_SEC;
			wait_time.tv_nsec     =       now_time.tv_usec*1000 + WAIT_USEC;//nano seconds
			{
				lockmutex lockm(&mutexData);
				while(upNum < upSum){
					timeoutret=pthread_cond_timedwait(&condUp,&mutexData,&wait_time);
					if(timeoutret==ETIMEDOUT){
						LOG(INFO)<<"time out " << upNum;
						break;
					}
				}
				if(upNum>0){
					layer->ComputeValueServer();
					pthread_cond_broadcast(&condCtrl);
				}
			}
		}
	}
template <typename Dtype>
	void Solver<Dtype>::ComputeValueServer(){
		ComputeUpdateValueServerThread();
		++itest;
		if(itest % param_.test_interval() ==0)
			TestAll();
		upNum=0;
	}
template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValueServerThread(){
	vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
	vector<float>& net_params_lr = this->net_->params_lr();
	vector<float>& net_params_weight_decay = this->net_->params_weight_decay();
	Dtype rate = GetLearningRate();//TODO iter_
	if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
		LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
	}
	Dtype momentum = this->param_.momentum();
	Dtype weight_decay = this->param_.weight_decay();
	string regularization_type = this->param_.regularization_type();
	for(int param_id = 0; param_id < net_params.size(); ++param_id){
		memset(net_params[param_id]->mutable_cpu_diff(),0,sizeof(Dtype)*(net_params[param_id]->count()));
	}
	for(int i=0;i<upSum;++i){
		if(flagCC[i]==1){
			Dtype **diff = ((Dtype***)tempDiff)[i];
			for(int param_id = 0; param_id < net_params.size(); ++param_id){
				caffe_axpy(net_params[param_id]->count(),(Dtype)1,
						&diff[param_id][0],net_params[param_id]->mutable_cpu_diff());
			}
		}
	}
	for (int param_id = 0; param_id < net_params.size(); ++param_id) {
		if(upNum!=0)caffe_scal(net_params[param_id]->count(),(Dtype)(1.0/upNum),net_params[param_id]->mutable_cpu_diff());
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
	this->net_->Update();
	for(int i=0;i<upSum;++i){
		if(flagCC[i]==1){
			//Dtype **data = ((Dtype***)tempData)[i]; //test del tempData20150113
			Dtype **diff = ((Dtype***)tempDiff)[i];
			for(int param_id = 0; param_id < net_params.size(); ++param_id){
				caffe_copy(net_params[param_id]->count(),
						net_params[param_id]->cpu_data(),
						&diff[param_id][0]);
			}
		}
	}
}
template <typename Dtype>
void* ComputeValueThreadClient(void* param) {
   SGDSolver<Dtype>* layer = static_cast<SGDSolver<Dtype>*>(((tprama*)param)->layer);
  int tid = ((tprama* )param)->tid;
  CHECK(layer);
  int flagFin=0;
    if(taskSum.getValue() <=0){ LOG(INFO)<<"client task out";pthread_exit(NULL);}
while(true){
    if(taskS.getValue()<taskS1)break;
    layer->ComputeValueClient(tid);
    sem_post(&semQ);
    pthread_mutex_lock(&mutexFin);
    if(taskSum.sub(1) <=0)flagFin=1;taskS.sub(1);
    pthread_cond_signal(&condFin);
    pthread_mutex_unlock(&mutexFin);
    if(flagFin)break;
}
LOG(INFO)<<"Thread fin "<<tid;
  return NULL;
}
template <typename Dtype>
void Solver<Dtype>::ComputeValueClient(int tid){
	int mpi_source;
	ComputeUpdateValueClientThread(mpi_source,tid);
	{
		lockmutex lockm(&mutexData);
		while(upNum!=0){
			pthread_cond_wait(&condCtrl,&mutexData);
			break;
		}
		flagCC[tid]=0;
	}
	//Dtype **data = ((Dtype***)tempData)[tid]; //test del tempData20150113
	Dtype **diff = ((Dtype***)tempDiff)[tid];
#if 0
	vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
	for (int param_id = 0; param_id < net_params.size(); ++param_id) {
		caffe_mpi_send<Dtype>(diff[param_id],net_params[param_id]->count(),
				mpi_source,TAG_NET_OUT,MPI_COMM_WORLD);
	}
#else
	caffe_mpi_send(diff[0],1,mpiTypeDiff,mpi_source,TAG_NET_OUT,MPI_COMM_WORLD);
#endif
	idleQ.push(mpi_source);
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
	delete [] flagCC;
#if 0
	if(this->rank==0){
		tempDiff=new Dtype**[upSum];
		for(int i=0;i<upSum;++i){
			delete[] ((Dtype***)tempDiff)[i][0];
			delete[] ((Dtype***)tempDiff)[i];
		}
		delete[] (Dtype***)tempDiff;
		if(this->mpiTypeDiff != MPI_DATATYPE_NULL)
			MPI_Type_free(&this->mpiTypeDiff);
	}else{
		if(this->mpiTypeCpuDiff != MPI_DATATYPE_NULL)
			MPI_Type_free(&this->mpiTypeCpuDiff);
		if(this->mpiTypeCpuData != MPI_DATATYPE_NULL)
			MPI_Type_free(&this->mpiTypeCpuData);
	}
#endif
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
	mpiTypeDiff = MPI_DATATYPE_NULL;
	mpiTypeCpuDiff = MPI_DATATYPE_NULL;
	mpiTypeCpuData = MPI_DATATYPE_NULL;
  if(rank==0){
	  if(idleQ.empty()){
		  for(int i=1;i<size;++i){
			  idleQ.push(i);
		  }
	  }
  }
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
	if(rank==0){
		pthread_mutex_init(&mutexFin,NULL);
		pthread_cond_init(&condFin,NULL);
		sem_init(&semQ,0,idleQ.size());
		taskSum.add(param_.max_iter()-iter_);
		int msize;
		int tNetCount=0;
		MPI_Comm_size (MPI_COMM_WORLD, &msize);
		upSum= msize -1 ;
		taskS.add(taskSum.getValue());
		taskS1=upSum;
		flagCC=new int[upSum];
		memset(flagCC,0,sizeof(int)*upSum);
		//tempData=new Dtype**[upSum];  //test del tempData20150113
		tempDiff=new Dtype**[upSum];
		for(int j=0;j<net_params.size();++j){
			tNetCount += net_params[j]->count();
		}
		for(int i=0;i<upSum;++i){
			//((Dtype***)tempData)[i]=new Dtype*[net_params.size()]; //test del tempData20150113
			((Dtype***)tempDiff)[i]=new Dtype*[net_params.size()];
#if 1
			((Dtype***)tempDiff)[i][0] = new Dtype[tNetCount];
			for(int j=1;j<net_params.size();++j){
				((Dtype***)tempDiff)[i][j]= ((Dtype***)tempDiff)[i][j-1]+net_params[j-1]->count();
			}
#else
			for(int j=0;j<net_params.size();++j){
				((Dtype***)tempDiff)[i][j]=new Dtype[net_params[j]->count()];
			}
#endif
		}
#if 1	
		MPI_Datatype *netDataType=new MPI_Datatype[net_params.size()];
		int *blocklen = new int[net_params.size()];
		MPI_Aint *displacement = new MPI_Aint[net_params.size()];
		Dtype **diff = ((Dtype***)tempDiff)[0];
		for (int param_id = 0; param_id < net_params.size(); ++param_id) {
			blocklen[param_id]=net_params[param_id]->count();
			if(typeid(Dtype)==typeid(float))
				netDataType[param_id] = MPI_FLOAT;
			else if(typeid(Dtype)==typeid(double))
				netDataType[param_id] = MPI_DOUBLE;
			else
				LOG(FATAL)<<"This datetype is not support!"<<typeid(Dtype).name();
			displacement[param_id] = (char*) diff[param_id]- (char*) diff[0];
		}
		MPI_Type_struct(net_params.size(),blocklen,displacement,netDataType,&mpiTypeDiff);
		MPI_Type_commit(&mpiTypeDiff);
		delete[] netDataType;
		delete[] blocklen;
		delete[] displacement;

#endif
		pthread_t threads;
		pthread_t *threadc=new pthread_t[msize-1];
		tprama pramas;
		pramas.layer=static_cast<void*>(this);
		pramas.tid=-1;
		CHECK(!pthread_create(&threads, NULL, ComputeValueThreadServer<Dtype>,
					&pramas)) << "Pthread(solve) execution failed.";
		tprama *pramac = new tprama[msize-1];
		for(int i=0;i<upSum;++i){
			pramac[i].layer = static_cast<void*>(this);
			pramac[i].tid = i;
			CHECK(!pthread_create(&threadc[i], NULL, ComputeValueThreadClient<Dtype>,
						&pramac[i])) << "Pthread(solve) execution failed.";
		}

		for (; iter_ < param_.max_iter(); ++iter_) {
			sem_wait(&semQ);
			if(!idleQ.empty()){
				caffe_mpi_send(&iter_,1,MPI_INT,idleQ.front(),TAG_ITER,MPI_COMM_WORLD);
				/*Dtype loss = */net_->ForwardBackwardRoot(bottom_vec,idleQ.front());
				idleQ.pop();
			}else{
				LOG(FATAL)<<"ERROR! idleQ is empty!";
			}
		}
		pthread_mutex_lock(&mutexFin);
		while(taskSum.getValue()>0){
			pthread_cond_wait(&condFin,&mutexFin);
			LOG(INFO)<<"TaskSum "<<taskSum.getValue();
		}
		pthread_mutex_unlock(&mutexFin);
		pthread_mutex_destroy(&mutexFin);
		pthread_cond_destroy(&condFin);
		while(!idleQ.empty()){
			int flagFin= -1;
			caffe_mpi_send(&flagFin,1,MPI_INT,idleQ.front(),TAG_ITER,MPI_COMM_WORLD);
			idleQ.pop();
		}
		TestAll();
		sleep(WAIT_SEC);
		for(int i=0;i<upSum;++i){
			pthread_cancel(threadc[i]);
		}
		pthread_cancel(threads);
		delete[] threadc;
		delete[] pramac;
		LOG(INFO)<<"DESTROY "<< (pthread_mutex_destroy(&mutexData));
	}else{
		int flagType=0;
		while(true){
			MPI_Status status;
			status.MPI_ERROR=0;
			caffe_mpi_recv(&iter_,1,MPI_INT,0,TAG_ITER,MPI_COMM_WORLD,&status);
			if(iter_== -1)break;
			net_->taskiter = iter_;
			Dtype loss = net_->ForwardBackward(bottom_vec);
			if(flagType==0){
				MPI_Datatype *netDataType=new MPI_Datatype[net_params.size()];
				int *blocklen = new int[net_params.size()];
				MPI_Aint *displacement = new MPI_Aint[net_params.size()];
				for (int param_id = 0; param_id < net_params.size(); ++param_id) {
					blocklen[param_id]=net_params[param_id]->count();
					if(typeid(Dtype)==typeid(float))
						netDataType[param_id] = MPI_FLOAT;
					else if(typeid(Dtype)==typeid(double))
						netDataType[param_id] = MPI_DOUBLE;
					else
						LOG(FATAL)<<"This datetype is not support!"<<typeid(Dtype).name();
					displacement[param_id] = (char*)(net_params[param_id]->mutable_cpu_diff()) - (char*)(net_params[0]->mutable_cpu_diff());
				}
				MPI_Type_struct(net_params.size(),blocklen,displacement,netDataType,&mpiTypeCpuDiff);
				MPI_Type_commit(&mpiTypeCpuDiff);
				for (int param_id = 0; param_id < net_params.size(); ++param_id) {
					displacement[param_id] = (char*)(net_params[param_id]->mutable_cpu_data()) - (char*)(net_params[0]->mutable_cpu_data());
				}
				MPI_Type_struct(net_params.size(),blocklen,displacement,netDataType,&mpiTypeCpuData);
				MPI_Type_commit(&mpiTypeCpuData);
				delete[] netDataType;
				delete[] blocklen;
				delete[] displacement;
				flagType=1;
			}
			ComputeUpdateValueClient();
			memset(&status,0,sizeof(status));
			vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
#if 0
			for (int param_id = 0; param_id < net_params.size(); ++param_id) {
				caffe_mpi_recv<Dtype>(net_params[param_id]->mutable_cpu_data(),net_params[param_id]->count(),
						0,TAG_NET_OUT,MPI_COMM_WORLD,&status);
			}
#else
			for (int param_id = 0; param_id < net_params.size(); ++param_id) {
				net_params[param_id]->mutable_cpu_data();
			}
			caffe_mpi_recv(net_params[0]->mutable_cpu_data(),1,
					mpiTypeCpuData,0,TAG_NET_OUT,MPI_COMM_WORLD,&status);
#endif
			// Save a snapshot if needed.
			if (param_.snapshot() && iter_ > start_iter &&
					iter_ % param_.snapshot() == 0) {
				Snapshot();//TODO
			}

			//if (param_.test_interval() && iter_ % param_.test_interval() == 0
			//		&& (iter_ > 0 || param_.test_initialization())) {
			//	TestAll();
			//}

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
	}
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
		LOG(INFO) << "Optimization Done.";
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
  vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  Dtype rate = GetLearningRate();
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
  }
#if 0
  for (int param_id = 0; param_id < net_params.size(); ++param_id) {
    if(param_id==0)
      caffe_mpi_send<Dtype>(net_params[param_id]->mutable_cpu_diff(),net_params[param_id]->count(),
	  0,TAG_UPDATE_1,MPI_COMM_WORLD);
    else
      caffe_mpi_send<Dtype>(net_params[param_id]->mutable_cpu_diff(),net_params[param_id]->count(),
	  0,TAG_UPDATE,MPI_COMM_WORLD);
  }
#else
  for (int param_id = 0; param_id < net_params.size(); ++param_id) {
		net_params[param_id]->mutable_cpu_diff();
  }
  caffe_mpi_send(net_params[0]->mutable_cpu_diff(),1,this->mpiTypeCpuDiff,0,TAG_UPDATE,MPI_COMM_WORLD);
#endif
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

        Dtype **diff = ((Dtype***)tempDiff)[tid];
#if 0
        vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
	for (int param_id = 0; param_id < net_params.size(); ++param_id) {
                memset(&status,0,sizeof(status));
                if(param_id==0){
                        caffe_mpi_recv<Dtype>(&diff[param_id][0],net_params[param_id]->count(),
                                        MPI_ANY_SOURCE,TAG_UPDATE_1,MPI_COMM_WORLD,&status);
                        mpi_source=status.MPI_SOURCE;
                }else{
                        caffe_mpi_recv<Dtype>(&diff[param_id][0],net_params[param_id]->count(),
                                        mpi_source,TAG_UPDATE,MPI_COMM_WORLD,&status);
                }
        }
#else
	caffe_mpi_recv(&diff[0][0],1,this->mpiTypeDiff,MPI_ANY_SOURCE,TAG_UPDATE,MPI_COMM_WORLD,&status);
	mpi_source=status.MPI_SOURCE;
#endif
#if 0
	int packsize,bufsize,position;
	char* packbuf;
	for (int param_id = 0; param_id < net_params.size(); ++param_id) {
		MPI_Pack(net_params[param_id]->mutable_cpu_data(),
			net_params[param_id]->count(),
			MPI_FLOAT,packbuf,bufsize,&packsize,MPI_COMM_WORLD);
	}
	for (int param_id = 0; param_id < net_params.size(); ++param_id) {
		MPI_Unpack(packbuf,packsize,&position,net_params[param_id]->mutable_cpu_data(),
			net_params[param_id]->count(),MPI_FLOAT,MPI_COMM_WORLD);
	}
#endif
	
{
        lockmutex lockm(&mutexData);
        flagCC[tid] = 1;
        ++upNum;
        pthread_cond_broadcast(&condUp);
}
}

INSTANTIATE_CLASS(Solver);
INSTANTIATE_CLASS(SGDSolver);
INSTANTIATE_CLASS(NesterovSolver);
INSTANTIATE_CLASS(AdaGradSolver);

}  // namespace caffe
