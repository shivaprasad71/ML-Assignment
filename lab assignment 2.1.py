import tensorflow as tf
import matplotlib.pyplot as plt
#
a = tf.constant([12.0,12.0,13.0,15.0,16.0,13.0,17.0],name ='x')
training_epoch = 500
learning_rate = 0.005
t1=0.0
t0=0.0
for epoch in range(training_epoch):
    h0=tf.divide(1.0,tf.add(1.0,tf.exp(-tf.add(t0,tf.reduce_sum(tf.multiply(t1,a))))))
    tn1 = tf.subtract(t1,tf.multiply(learning_rate,tf.divide(h0,tf.cast(tf.size(a),tf.float32))))
    t1=tn1
    model=tf.global_variables_initializer()
sess = tf.Session() 
h0=tf.cond(h0<0.5,lambda:0.00,lambda:1.00)
print("h0 value:",sess.run(h0))

a = tf.constant([12.0,12.0,13.0,15.0,16.0,13.0,17.0],name ='x')
training_epoch = 500
learning_rate = 0.005
t1=0.0
t0=0.0
for epoch in range(training_epoch):
    h0=tf.divide(1.0,tf.add(1.0,tf.exp(-tf.add(t0,tf.reduce_sum(tf.multiply(t1,a))))))
    tn1 = tf.subtract(t1,tf.multiply(learning_rate,tf.divide(h0,tf.cast(tf.size(a),tf.float32))))
    t1=tn1
    model=tf.global_variables_initializer()
sess = tf.Session() 
h0=tf.cond(h0<0.5,lambda:0.00,lambda:1.00)
print("h0 value:",sess.run(h0))

a = tf.constant([12.0,12.0,13.0,15.0,16.0,13.0,17.0],name ='x')
training_epoch = 500
learning_rate = 0.005
t1=0.0
t0=0.0
for epoch in range(training_epoch):
    h0=tf.divide(1.0,tf.add(1.0,tf.exp(-tf.add(t0,tf.reduce_sum(tf.multiply(t1,a))))))
    tn1 = tf.subtract(t1,tf.multiply(learning_rate,tf.divide(h0,tf.cast(tf.size(a),tf.float32))))
    t1=tn1
    model=tf.global_variables_initializer()
sess = tf.Session() 
h0=tf.cond(h0<0.5,lambda:0.00,lambda:1.00)
print("h0 value:",sess.run(h0))
