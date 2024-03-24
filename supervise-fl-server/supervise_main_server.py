import socketserver
import pickle, struct
import os
import sys
import argparse
import time
from threading import Lock, Thread
import threading
import numpy as np


# np.set_printoptions(threshold=np.inf)


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    ## FL
    parser.add_argument('--fl_round', type=int, default=10,
                    help='communication to server after the epoch of local training')
    parser.add_argument('--num_of_users', type=int, default=4,##8 for audio
                    help='num of users in FL')

    ## system
    parser.add_argument('--start_wait_time', type=int, default=300,
                    help='start_wait_time')
    parser.add_argument('--W_wait_time', type=int, default=7200,
                    help='W_wait_time')
    parser.add_argument('--end_wait_time', type=int, default=7200,
                    help='end_wait_time')

    ## model
    parser.add_argument('--dim_weight_supervise', type=int, default = 41659216)

    opt = parser.parse_args()

    return opt


opt = parse_option()


iteration_count = 0
trial_count = 0
NUM_OF_WAIT = opt.num_of_users

## recieved all model weights
weight_COLLECTION = np.zeros((opt.num_of_users, opt.dim_weight_supervise))
weight_MEAN = np.zeros(opt.dim_weight_supervise)

Update_Flag = np.ones(opt.num_of_users)
conver_indicator = 1e5

wait_time_record = np.zeros(opt.fl_round)
aggregation_time_record = np.zeros(opt.fl_round)
downlink_time_record = np.zeros((opt.num_of_users, opt.fl_round))
server_start_time_record = np.zeros((opt.num_of_users, opt.fl_round))


def mmFedavg(opt, model_weight):

	mean_model_weight = np.mean(model_weight, axis = 0)

	return mean_model_weight


def server_update():
	
	global opt, iteration_count, weight_COLLECTION, weight_MEAN
	global aggregation_time_record, wait_time_record, server_start_time_record

	aggregate_time1 = time.time()
	wait_time_record[iteration_count] = aggregate_time1 - np.min(server_start_time_record[:, iteration_count])
	print("server wait time:", wait_time_record[iteration_count])

	## mmFedavg for model weights
	print("Iteration {}: mmFedavg of model weights".format(iteration_count))
	weight_MEAN = mmFedavg(opt, weight_COLLECTION)


	aggregate_time2 = time.time()
	aggregation_time_record[iteration_count] = aggregate_time2 - aggregate_time1
	print("server aggregation time:", aggregation_time_record[iteration_count])

	iteration_count = iteration_count + 1
	print("iteration_count: ", iteration_count)

	
def reinitialize():

	global iteration_count
	# trial_count += 1
	iteration_count = 0
	# print("Trial: ", trial_count)

	global opt, NUM_OF_WAIT, wait_time_record, aggregation_time_record, server_start_time_record, downlink_time_record
	print("All of Server Wait Time:", np.sum(wait_time_record))
	print("All of Server Aggregate Time:", np.sum(aggregation_time_record))

	save_model_path = "./save_server_time_supervise_{}nodes/".format(opt.num_of_users)
	if not os.path.isdir(save_model_path):
		os.makedirs(save_model_path)

	np.savetxt(os.path.join(save_model_path, "aggregation_time_record.txt"), aggregation_time_record)
	np.savetxt(os.path.join(save_model_path, "wait_time_record.txt"), wait_time_record)
	np.savetxt(os.path.join(save_model_path, "server_start_time_record.txt"), server_start_time_record)
	np.savetxt(os.path.join(save_model_path, "downlink_time_record.txt"), downlink_time_record)

	wait_time_record = np.zeros(opt.fl_round)
	aggregation_time_record = np.zeros(opt.fl_round)
	server_start_time_record = np.zeros((opt.num_of_users, opt.fl_round))
	downlink_time_record = np.zeros((opt.num_of_users, opt.fl_round))

	opt = parse_option()
	NUM_OF_WAIT = opt.num_of_users

	global weight_COLLECTION, Update_Flag

	weight_COLLECTION = np.zeros((opt.num_of_users, opt.dim_weight_supervise))
	Update_Flag = np.ones(opt.num_of_users)

	global weight_MEAN
	weight_MEAN = np.zeros(opt.dim_weight_supervise)
	
	barrier_update()


barrier_start = threading.Barrier(NUM_OF_WAIT,action = None, timeout = None)
barrier_W = threading.Barrier(NUM_OF_WAIT,action = server_update, timeout = None)
barrier_end = threading.Barrier(NUM_OF_WAIT, action = reinitialize, timeout = None)

def barrier_update():
	global NUM_OF_WAIT
	print("update the barriers to NUM_OF_WAIT: ",NUM_OF_WAIT)
	global barrier_W
	barrier_W = threading.Barrier(NUM_OF_WAIT,action = server_update, timeout = None)
	global barrier_end
	barrier_end = threading.Barrier(NUM_OF_WAIT, action = reinitialize, timeout = None)


class MyTCPHandler(socketserver.BaseRequestHandler):

	def send2node(self, var):

		var_data = pickle.dumps(var, protocol = 0)
		var_size = sys.getsizeof(var_data)
		var_header = struct.pack("i",var_size)
		self.request.sendall(var_header)
		self.request.sendall(var_data)

		return var_size


	def handle(self):
		while True:
			try:
				#receive the size of content
				header = self.request.recv(4)
				size = struct.unpack('i', header)

				#receive the id of client
				u_id = self.request.recv(4)
				temp_id = struct.unpack('i',u_id)

				user_id = int(temp_id[0])
				print("user_id:", user_id)

				# receive the type of message, defination in communication.py
				mess_type = self.request.recv(4)
				mess_type = struct.unpack('i',mess_type)[0]

				#print("This is the {}th node with message type {}".format(user_id[0],mess_type))

				#receive the body of message
				recv_data = b""
				
				while sys.getsizeof(recv_data)<size[0]:
					recv_data += self.request.recv(size[0]-sys.getsizeof(recv_data))
				
				#if hello message, barrier until all clients arrive and send a message to start
				if mess_type == -1:
					try:
						barrier_start.wait(opt.start_wait_time)
					except Exception as e:
						print("start wait timeout...")
						print(e)

					start_message = 'start'
					mess_size = self.send2node(start_message)


				# if modality message, record the local modality
				elif mess_type == 1:

					try:
						barrier_start.wait(opt.start_wait_time)
					except Exception as e:
						print("wait W timeout...")
						print(e)

					temp_modality = pickle.loads(recv_data)

					print("client {} has modality {}".format(user_id, temp_modality))


				#if W message, server update for model aggregation
				elif mess_type == 0:

					server_start_time_record[user_id, iteration_count] = time.time()

					weights = pickle.loads(recv_data)
					# print(weights.shape)

					weight_COLLECTION[user_id] = weights

					print("Round {}: received weight from client {}, with shape {} ".format(iteration_count, user_id, weights.shape))
					# print("error 1")

					try:
						barrier_W.wait(opt.W_wait_time)
					except Exception as e:
						print("wait W timeout...")
						print(e)
						
					send_weight = weight_MEAN

					# print("error 3")
					downlink_time1 = time.time()
					mess_size = self.send2node(send_weight)
					downlink_time2 = time.time()
					model_downlink_time = downlink_time2 - downlink_time1
					downlink_time_record[user_id, iteration_count] = model_downlink_time
					print("send New_W to client {} with the size of {}, shape of {}: time {}".format(user_id, mess_size, send_weight.shape, model_downlink_time))


					#if Update_Flag=0, stop the specific client
					if Update_Flag[user_id]==0:
						
						sig_stop = struct.pack("i",2)

						global NUM_OF_WAIT
						NUM_OF_WAIT-=1
						barrier_update()
						self.finish()

					#if convergence, stop all the clients
					elif(np.abs(conver_indicator)<1e-2 or (iteration_count == opt.fl_round)):
						sig_stop = struct.pack("i",1)
					else:
						sig_stop = struct.pack("i",0)
					self.request.sendall(sig_stop)


				elif mess_type == 9:
					break

				elif mess_type == 10:
					try:
						barrier_end.wait(opt.end_wait_time)
					except Exception as e:
						print("finish timeout...")
					break


			except Exception as e:
				print('err',e)
				break



if __name__ == "__main__":
	# HOST, PORT = "0.0.0.0", 9998 

	HOST = "0.0.0.0"#127.0.0.1

	port_num = 30415

	server = socketserver.ThreadingTCPServer((HOST,port_num),MyTCPHandler)
	# server.server_close()
	server.serve_forever(poll_interval = 0.5)
