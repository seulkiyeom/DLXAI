'''
@author: Vignesh Srinivasan
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Vignesh Srinivasan
@maintainer: Sebastian Lapuschkin
@contact: vignesh.srinivasan@hhi.fraunhofer.de
@date: 20.12.2016
@version: 1.0+
@copyright: Copyright (c) 2016-2017, Vignesh Srinivasan, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

import tensorflow as tf
from module import Module


from math import ceil

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

import numpy as np


class MaxPool(Module):

	def __init__(self, pool_size=2, pool_stride=None, pad = 'SAME',name='maxpool'):
		self.name = name
		Module.__init__(self)
		self.pool_size = pool_size
		#self.pool_kernel = [1, self.pool_size, self.pool_size, 1]
		self.pool_kernel = [1] + [self.pool_size, self.pool_size, self.pool_size] + [1]
		#self.strides = pool_stride
		self.strides = pool_stride
		if self.strides is None:
			self.stride_size=self.pool_size
			#self.strides=[1, self.stride_size, self.stride_size, 1]
			self.strides = [1, self.stride_size, self.stride_size, self.stride_size, 1]
		self.pad = pad

	def forward(self, input_tensor, batch_size=10, img_dim=28):
		self.input_tensor = input_tensor
		#self.in_N, self.in_h, self.in_w, self.input_depth = self.input_tensor.get_shape().as_list()
		self.in_N, self.input_x_dim, self.input_y_dim, self.input_z_dim, self.input_depth = self.input_tensor.get_shape().as_list()
		#with tf.variable_scope(self.name):
		with tf.name_scope(self.name):
			#self.activations = tf.nn.max_pool(self.input_tensor, ksize=self.pool_kernel,strides=self.strides,padding=self.pad, name=self.name )
			self.activations = tf.nn.max_pool3d(self.input_tensor, ksize=self.pool_kernel, strides=self.strides,
												padding=self.pad, name=self.name)
			tf.summary.histogram('activations', self.activations)

		return self.activations

	def clean(self):
		self.activations = None
		self.R = None


	'''def _simple_lrp(self,R):
		#LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
		self.check_shape(R)
		image_patches = self.extract_patches()
		return self.restitch_image(image_patches)
	'''

	def _simple_lrp(self,R):
		import time; start_time = time.time()
		
		self.R = R
		#R_shape = self.R.get_shape().as_list()
		#activations_shape = self.activations.get_shape().as_list()
		#if len(R_shape)!=4:
		#	self.R = tf.reshape(self.R, activations_shape)

		N,xout,yout,zout,NF = self.R.get_shape().as_list()
		_, xstride, ystride, zstride, _ = self.strides
		#out_N, out_h, out_w, out_depth = self.activations.get_shape().as_list()
		in_N, in_x, in_y, in_z, in_depth = self.input_tensor.get_shape().as_list()

		xf,yf,zf,df,NF = [self.pool_size, self.pool_size, self.pool_size, in_depth, NF]
		


		if self.pad == 'SAME':

			#print(xout, xstride, xf, in_x)
			#print(zout, zstride, zf, in_z)
			px = (xout -1) * xstride + xf - in_x
			py = (yout -1) * ystride + yf - in_y
			pz = (zout -1) * zstride + zf - in_z

			px = 0 if (px < 0) else px
			py = 0 if (py < 0) else py
			pz = 0 if (pz < 0) else pz
			

			# pr = (out_h -1) * hstride + hf - in_h
			# pc =  (out_w -1) * wstride + wf - in_w
			p_top = px/2
			p_bottom = px-(px/2)
			p_left = py/2
			p_right = py-(py/2)
			p_front = pz/2
			p_back = pz-(pz/2)
			self.pad_input_tensor = tf.pad(self.input_tensor, [[0,0],[p_top,p_bottom],[p_left, p_right],[p_front,p_back],[0,0]], "CONSTANT")
		elif self.pad == 'VALID':
			self.pad_input_tensor = self.input_tensor
			
		pad_in_N, pad_in_x, pad_in_y, pad_in_z, pad_in_depth = self.pad_input_tensor.get_shape().as_list()
		Rx = tf.zeros_like(self.pad_input_tensor, dtype = tf.float32)
		
		#pdb.set_trace()
		weights = tf.ones([self.pool_size, self.pool_size, self.pool_size, df, NF])
		term1 = tf.expand_dims(weights, 0)
		#t2 = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(self.biases, 0), 0), 0),0)
		for i in xrange(xout):
			for j in xrange(yout):
				for k in xrange(zout):
					input_slice = self.pad_input_tensor[:, i*xstride:i*xstride+xf , j*ystride:j*ystride+yf , k*zstride:k*zstride+zf , : ]
					term2 =  tf.expand_dims(input_slice, -1)
					#pdb.set_trace()
					Z = term1 * term2
					t1 = tf.reduce_sum(Z, [1,2,3,4], keep_dims=True)
					#Zs = t1 + t2
					Zs = t1
					stabilizer = 1e-8*(tf.where(tf.greater_equal(Zs,0), tf.ones_like(Zs, dtype=tf.float32), tf.ones_like(Zs, dtype=tf.float32)*-1))
					Zs += stabilizer
					result = tf.reduce_sum((Z/Zs) * tf.expand_dims(self.R[:,i:i+1,j:j+1,k:k+1,:], 3), 5)
				
					#pdb.set_trace()
					#pad each result to the dimension of the out
					pad_bottom = pad_in_x - (i*xstride+xf) if( pad_in_x - (i*xstride+xf))>0 else 0
					pad_top = i*xstride
					pad_right = pad_in_y - (j*ystride+yf) if ( pad_in_y - (j*ystride+yf) > 0) else 0
					pad_left = j*ystride
					
					pad_front = pad_in_z - (k*zstride+zf) if ( pad_in_z - (k*zstride+zf) > 0) else 0
					pad_back = k*zstride

					result = tf.pad(result, [[0,0],[pad_top, pad_bottom],[pad_left, pad_right],[pad_front,pad_back],[0,0]], "CONSTANT")
					# print(i,j)
					# print(i*hstride, i*hstride+hf , j*wstride, j*wstride+wf)
					# print(pad_top, pad_bottom,pad_left, pad_right)
					Rx+= result
		#pdb.set_trace()
		total_time = time.time() - start_time
		print(total_time)
		if self.pad=='SAME':
			return Rx[:, (px/2):in_x+(px/2), (py/2):in_y+(py/2),(pz/2):in_z+(pz/2), :]
		elif self.pad =='VALID':
			return Rx


	def _epsilon_lrp(self,R, epsilon):
		'''
		Since there is only one (or several equally strong) dominant activations, default to _simple_lrp
		'''
		return self._simple_lrp(R)

	def _ww_lrp(self,R): 
		'''
		There are no weights to use. default to _flat_lrp(R)
		'''
		return self._flat_lrp(R)
	
	def _flat_lrp(self,R):
		'''
		distribute relevance for each output evenly to the output neurons' receptive fields.
		'''
		self.check_shape(R)

		Z = tf.ones([self.in_N, self.Hout,self.Wout, self.pool_size,self.pool_size, self.input_depth])
		Zs = self.compute_zs(Z)
		result = self.compute_result(Z,Zs)
		return self.restitch_image(result)
	
	def _alphabeta_lrp(self,R,alpha):
		'''
		Since there is only one (or several equally strong) dominant activations, default to _simple_lrp
		'''
		return self._simple_lrp(R)
	
	def check_shape(self, R):
		self.R = R
		R_shape = self.R.get_shape().as_list()
		activations_shape = self.activations.get_shape().as_list()
		if len(R_shape)!=4:
			self.R = tf.reshape(self.R, activations_shape)
		#N,self.Hout,self.Wout,NF = self.R.get_shape().as_list()


	def extract_patches(self):
		try:
			self.dims = self.R.get_shape().as_list()
		except:
			self.dims = self.R.shape

		dim = [self.input_x_dim + self.pool_size, self.input_y_dim + self.pool_size, self.input_z_dim + self.pool_size]

		input_tensor = tf.pad(self.input_tensor, [[0,0],[0,self.pool_size-1],[0,self.pool_size-1],[0,self.pool_size-1],[0,0]],'CONSTANT')

		#output = np.zeros([self.in_N, dim[0]/self.stride_size, dim[1]/self.stride_size, dim[2]/self.stride_size, self.input_depth], dtype=object)


		output = [[[[[[tf.slice(input_tensor, [nth_layer, x, y, z, nth_depth], [1, self.pool_size, self.pool_size, self.pool_size, 1])] for nth_layer in range(0, self.batch_size)] for x in range(0, self.dims[1])] for y in range(0, self.dims[2])] for z in range(0, self.dims[3])] for nth_depth in range(0, self.input_depth)]

		return tf.reshape(output, [self.in_N, self.dims[1], self.dims[2], self.dims[3], self.pool_size*self.pool_size*self.pool_size*self.input_depth])


		
	# def _simple__lrp(self,R):
	#	 '''
	#	 LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
	#	 '''
	#	 import time; start_time = time.time()
		
	#	 self.R = R
	#	 R_shape = self.R.get_shape().as_list()
	#	 activations_shape = self.activations.get_shape().as_list()
	#	 if len(R_shape)!=4:
	#		 self.R = tf.reshape(self.R, activations_shape)

	#	 N,Hout,Wout,NF = self.R.get_shape().as_list()
	#	 _, hf,wf,_ = self.pool_size
	#	 _, hstride, wstride, _ = self.strides
	#	 in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

	#	 # op1 = tf.extract_image_patches(self.input_tensor, ksizes=[1, hf,wf, 1], strides=[1, hstride, wstride, 1], rates=[1, 1, 1, 1], padding=self.pad)
	#	 # p_bs, p_h, p_w, p_c = op1.get_shape().as_list()
	#	 # image_patches = tf.reshape(op1, [p_bs,p_h,p_w, hf, wf, in_depth])

	#	 # op1 = tf.extract_image_patches(self.input_tensor, ksizes=[1, hf,wf, 1], strides=[1, hstride, wstride, 1], rates=[1, 1, 1, 1], padding=self.pad)
	#	 # p_bs, p_h, p_w, p_c = op1.get_shape().as_list()
	#	 image_patches = tf.reshape(tf.extract_image_patches(self.input_tensor, ksizes=[1, hf,wf, 1], strides=[1, hstride, wstride, 1], rates=[1, 1, 1, 1], padding=self.pad), [N,Hout,Wout, hf, wf, in_depth])

	#	 #import pdb; pdb.set_trace()
	#	 Z = tf.equal( tf.reshape(self.activations, [N,Hout,Wout,1,1,NF]), image_patches)
	#	 Z = tf.where(Z, tf.ones_like(Z, dtype=tf.float32), tf.zeros_like(Z,dtype=tf.float32) )
	#	 #Z = tf.expand_dims(self.weights, 0) * tf.expand_dims( image_patches, -1)
	#	 Zs = tf.reduce_sum(Z, [3,4,5], keep_dims=True)  #+ tf.expand_dims(self.biases, 0)
	#	 stabilizer = 1e-12*(tf.where(tf.greater_equal(Zs,0), tf.ones_like(Zs, dtype=tf.float32), tf.ones_like(Zs, dtype=tf.float32)*-1))
	#	 Zs += stabilizer
	#	 #result =   (Z/Zs) * tf.reshape(self.R, [in_N,Hout,Wout,1,1,NF])
	#	 total_time = time.time() - start_time
	#	 print(total_time)
	#	 return self.patches_to_images(tf.reshape( (Z/Zs) * tf.reshape(self.R, [in_N,Hout,Wout,1,1,NF]), [N,Hout,Wout, hf*wf*in_depth]), in_N, in_h, in_w, in_depth, Hout, Wout, hf,wf, hstride,wstride )
		
	#	 #return Rx

	# def __simple_lrp(self,R):
	#	 '''
	#	 LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
	#	 '''
	#	 import time; start_time = time.time()
			
	#	 self.R = R
	#	 R_shape = self.R.get_shape().as_list()
	#	 if len(R_shape)!=4:
	#		 activations_shape = self.activations.get_shape().as_list()
	#		 self.R = tf.reshape(self.R, activations_shape)
			

		
	#	 N,Hout,Wout,NF = self.R.get_shape().as_list()
	#	 _,hf,wf,_ = self.pool_size
	#	 _,hstride, wstride,_= self.strides

	#	 out_N, out_rows, out_cols, out_depth = self.activations.get_shape().as_list()
	#	 in_N, in_rows, in_cols, in_depth = self.input_tensor.get_shape().as_list()
		
	#	 if self.pad == 'SAME':
	#		 pr = (Hout -1) * hstride + hf - in_rows
	#		 pc =  (Wout -1) * wstride + wf - in_cols
	#		 #similar to TF pad operation 
	#		 self.pad_input_tensor = tf.pad(self.input_tensor, [[0,0],[pr/2, (pr-(pr/2))],[pc/2,(pc - (pc/2))],[0,0]], "CONSTANT")
	#	 elif self.pad == 'VALID':
	#		 self.pad_input_tensor = self.input_tensor


	#	 pad_in_N, pad_in_rows, pad_in_cols, pad_in_depth = self.pad_input_tensor.get_shape().as_list()
		
	#	 Rx = tf.zeros_like(self.pad_input_tensor, dtype = tf.float32)
	#	 for i in xrange(Hout):
	#		 for j in xrange(Wout):
	#			 input_slice = self.pad_input_tensor[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : ]
	#			 Z = tf.equal( self.activations[:,i:i+1, j:j+1,:], input_slice)
	#			 Z = tf.where(Z, tf.ones_like(Z, dtype=tf.float32), tf.zeros_like(Z,dtype=tf.float32) )
	#			 Zs = tf.reduce_sum(Z, [1,2], keep_dims=True)
	#			 result = (Z/Zs) * self.R[:,i:i+1,j:j+1,:]
	#			 #pad each result to the dimension of the out
	#			 pad_right = pad_in_rows - (i*hstride+hf) if( pad_in_rows - (i*hstride+hf))>0 else 0
	#			 pad_left = i*hstride
	#			 pad_bottom = pad_in_cols - (j*wstride+wf) if ( pad_in_cols - (j*wstride+wf) > 0) else 0
	#			 pad_up = j*wstride
	#			 result = tf.pad(result, [[0,0],[pad_left, pad_right],[pad_up, pad_bottom],[0,0]], "CONSTANT")
				
	#			 Rx+= result
	#	 total_time = time.time() - start_time
	#	 print(total_time)
		
	#	 if self.pad=='SAME':
	#		 return Rx[:, (pc/2):in_cols+(pc/2), (pr/2):in_rows+(pr/2), :]
	#	 elif self.pad =='VALID':
	#		 return Rx
		
	# def _flat_lrp(self,R):
	#	 '''
	#	 distribute relevance for each output evenly to the output neurons' receptive fields.
	#	 '''

	#	 self.R = R
	#	 R_shape = self.R.get_shape().as_list()
	#	 if len(R_shape)!=4:
	#		 activations_shape = self.activations.get_shape().as_list()
	#		 self.R = tf.reshape(self.R, activations_shape)
			
	#	 N,Hout,Wout,NF = self.R.get_shape().as_list()
	#	 _,hf,wf,_ = self.pool_size
	#	 _,hstride, wstride,_= self.strides

	#	 out_N, out_rows, out_cols, out_depth = self.activations.get_shape().as_list()
	#	 in_N, in_rows, in_cols, in_depth = self.input_tensor.get_shape().as_list()

	#	 if self.pad == 'SAME':
	#		 pr = (Hout -1) * hstride + hf - in_rows
	#		 pc =  (Wout -1) * wstride + wf - in_cols
	#		 #similar to TF pad operation 
	#		 self.pad_input_tensor = tf.pad(self.input_tensor, [[0,0],[pr/2, (pr-(pr/2))],[pc/2,(pc - (pc/2))],[0,0]], "CONSTANT")
	#	 elif self.pad == 'VALID':
	#		 self.pad_input_tensor = self.input_tensor

	#	 pad_in_N, pad_in_rows, pad_in_cols, pad_in_depth = self.pad_input_tensor.get_shape().as_list()
		
	#	 Rx = tf.zeros_like(self.pad_input_tensor, dtype = tf.float32)
	#	 for i in xrange(Hout):
	#		 for j in xrange(Wout):
	#			 Z = tf.ones([N, hf,wf,NF], dtype=tf.float32)
	#			 Zs = tf.reduce_sum(Z, [1,2], keep_dims=True)
	#			 result = (Z/Zs) * self.R[:,i:i+1,j:j+1,:]
	#			 #pad each result to the dimension of the out
	#			 pad_bottom = pad_in_h - (i*hstride+hf) if( pad_in_h - (i*hstride+hf))>0 else 0
	#			 pad_top = i*hstride
	#			 pad_right = pad_in_w - (j*wstride+wf) if ( pad_in_w - (j*wstride+wf) > 0) else 0
	#			 pad_left = j*wstride
	#			 result = tf.pad(result, [[0,0],[pad_top, pad_bottom],[pad_left, pad_right],[0,0]], "CONSTANT")
				
	#			 Rx+= result
	#	 if self.pad=='SAME':
	#		 return Rx[:, (pc/2):in_w+(pc/2), (pr/2):in_h+(pr/2), :]
	#	 elif self.pad =='VALID':
	#		 return Rx		   



	def patches_to_images(self, grad, batch_size, rows_in, cols_in, channels, rows_out, cols_out, ksize_r, ksize_c, stride_h, stride_r ):
		rate_r = 1
		rate_c = 1
		padding = self.pad
		
		
		ksize_r_eff = ksize_r + (ksize_r - 1) * (rate_r - 1)
		ksize_c_eff = ksize_c + (ksize_c - 1) * (rate_c - 1)

		if padding == 'SAME':
			rows_out = int(ceil(rows_in / stride_r))
			cols_out = int(ceil(cols_in / stride_h))
			pad_rows = ((rows_out - 1) * stride_r + ksize_r_eff - rows_in) // 2
			pad_cols = ((cols_out - 1) * stride_h + ksize_c_eff - cols_in) // 2

		elif padding == 'VALID':
			rows_out = int(ceil((rows_in - ksize_r_eff + 1) / stride_r))
			cols_out = int(ceil((cols_in - ksize_c_eff + 1) / stride_h))
			pad_rows = (rows_out - 1) * stride_r + ksize_r_eff - rows_in
			pad_cols = (cols_out - 1) * stride_h + ksize_c_eff - cols_in

		pad_rows, pad_cols = max(0, pad_rows), max(0, pad_cols)

		grad_expanded = array_ops.transpose(
			array_ops.reshape(grad, (batch_size, rows_out,
									 cols_out, ksize_r, ksize_c, channels)),
			(1, 2, 3, 4, 0, 5)
		)
		grad_flat = array_ops.reshape(grad_expanded, (-1, batch_size * channels))

		row_steps = range(0, rows_out * stride_r, stride_r)
		col_steps = range(0, cols_out * stride_h, stride_h)

		idx = []
		for i in range(rows_out):
			for j in range(cols_out):
				r_low, c_low = row_steps[i] - pad_rows, col_steps[j] - pad_cols
				r_high, c_high = r_low + ksize_r_eff, c_low + ksize_c_eff

				idx.extend([(r * (cols_in) + c,
				   i * (cols_out * ksize_r * ksize_c) +
				   j * (ksize_r * ksize_c) +
				   ri * (ksize_c) + ci)
				  for (ri, r) in enumerate(range(r_low, r_high, rate_r))
				  for (ci, c) in enumerate(range(c_low, c_high, rate_c))
				  if 0 <= r and r < rows_in and 0 <= c and c < cols_in
				])

		sp_shape = (rows_in * cols_in,
			  rows_out * cols_out * ksize_r * ksize_c)

		sp_mat = sparse_tensor.SparseTensor(
			array_ops.constant(idx, dtype=ops.dtypes.int64),
			array_ops.ones((len(idx),), dtype=ops.dtypes.float32),
			sp_shape
		)

		jac = sparse_ops.sparse_tensor_dense_matmul(sp_mat, grad_flat)

		grad_out = array_ops.reshape(
			jac, (rows_in, cols_in, batch_size, channels)
		)
		grad_out = array_ops.transpose(grad_out, (2, 0, 1, 3))
		
		return grad_out
