#!/usr/bin/env sh
# args for EXTRACT_FEATURE

export PYTHONPATH=$PYTHONPATH:/home/s03/lyn/temporal-segment-networks/scripts/feature_extraction/
#ln -s /home/tian/lyn/DeconvNet-master/caffe/examples/sketch/snapshot_nyu/
#ln -s /home/tian/lyn/DeconvNet-master/caffe/data/
TOOL=../../lib/caffe-action/build/install/bin
MODEL=../../models/IsoGD_split1_tsn_depth_fc_2s_iter_50000.caffemodel #下载得到的caffe model
PROTOTXT=./tsn_bn_inception_rgb_extract_fc.prototxt # 网络定义
LAYER=fc6 # 提取层的名字，如提取fc7等
SUFFIX=val
LEVELDB=./_temp/$LAYER-$SUFFIX # 保存的leveldb路径
BATCHSIZE=1




# args for LEVELDB to MAT
DIM=747  #4096 # 需要手工计算feature长度

OUT=./_temp/$LAYER-$SUFFIX.mat #.mat文件保存路径
BATCHNUM=1 # 有多少个batch， 本例只有两张图， 所以只有一个batch

GLOG_logtostderr=1 $TOOL/extract_features  $MODEL $PROTOTXT $LAYER $LEVELDB $BATCHSIZE lmdb 

#python /home/s03/lyn/temporal-segment-networks/scripts/feature_extraction/lmdb2mat.py $LEVELDB $BATCHNUM  $BATCHSIZE $DIM $OUT
