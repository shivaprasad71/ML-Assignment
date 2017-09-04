
# coding: utf-8

# In[12]:

# Calculating mean
import tensorflow as tf
import numpy as np
x = tf.constant([1.5,2.3,4.2,5.6], shape =[2,2], name = "array_x")
y = tf.constant([2.5,3.3], shape =[2,2], name = "array_y")
m1 = tf.reduce_mean(x)
m2 = tf.reduce_mean(y)
sess = tf.Session()
print(sess.run(m1))
print(sess.run(m2))


# Variance
# variance(x)=sum((x−mean(x))2)
x = tf.constant([1.5,2.3,4.2,5.6], shape =[2,2], name = "array_x")
y = tf.constant([2.5,3.3], shape =[2,2], name = "array_y")
msq = tf.square(m1)
print(sess.run(msq))
a1 = tf.subtract(x[0,0],msq)
a2 = tf.subtract(x[0,1],msq)
a3 = tf.subtract(x[1,0],msq)
a4 = tf.subtract(x[1,1],msq)
print(sess.run(a1))
print(sess.run(a2))
print(sess.run(a3))
print(sess.run(a4))
s1 = tf.add(a1,a2)
s2 = tf.add(s1,a3)
s3 = tf.add(s2,a4)

s4 = tf.add(s1,s2)
sum = tf.add(s4,s3)
print(sess.run(sum))


# covariance=sum((x(i)−mean(x))∗(y(i)−mean(y)))
# for array X
b1 = tf.subtract(x[0,0],m1)
b2 = tf.subtract(x[0,1],m1)
b3 = tf.subtract(x[1,0],m1)
b4 = tf.subtract(x[1,1],m1)

su1 = tf.add(b1,b2)
su2 = tf.add(su1,b3)
su3 = tf.add(su2,b4)
su4 = tf.add(su1,su2)
sum1 = tf.add(su4,su3)

print(sess.run(sum1))

# For array Y

c1 = tf.subtract(y[0,0],m2)
c2 = tf.subtract(y[0,1],m2)
c3 = tf.subtract(y[1,0],m2)
c4 = tf.subtract(y[1,1],m2)

sm1 = tf.add(c1,c2)
sm2 = tf.add(sm1,c3)
sm3 = tf.add(sm2,c4)
sm4 = tf.add(sm1,sm2)
sum2 = tf.add(sm4,sm3)

print(sess.run(sum2))

# multiply 
prod = tf.multiply(sum1,sum2)
print(sess.run(prod))

# c=covariance(x,y)/variance(x)

c = tf.divide(prod,sum)
print(sess.run(c))


# m=mean(y)−c∗mean(x)

var = tf.multiply(c,m1)
m = tf.subtract(m2,var)
print(sess.run(m))


# In[ ]:



