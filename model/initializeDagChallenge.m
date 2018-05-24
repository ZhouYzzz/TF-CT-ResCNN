function net = initializeDagChallenge()
% INITIALIZE AN U-NET

net=dagnn.DagNN(); 

% The Dag network architecture;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                  STAGE I
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ----------------------------------
% Stage 1: 1st conv block
% ----------------------------------
conv1 = dagnn.Conv_original('size',[3,3,1,64], 'pad', 1, 'stride', 1, 'hasBias', true);
net.addLayer('conv1', conv1, {'FBP'},{'conv_x1'},{'conv_f1','conv_b1'});

net.addLayer('bn1', dagnn.BatchNorm('numChannels', 64), {'conv_x1'}, {'bn_x1'}, {'bn1f', 'bn1b', 'bn1m'});

relu1 = dagnn.ReLU();
net.addLayer('relu1', relu1, {'bn_x1'}, {'relu_x1'}, {});

% ----------------------------------
% Stage 1: 2nd conv block
% ----------------------------------
conv2 = dagnn.Conv_original('size',[3,3,64,64], 'pad', 1, 'stride', 1, 'hasBias', true);
net.addLayer('conv2', conv2, {'relu_x1'},{'conv_x2'},{'conv_f2','conv_b2'});

net.addLayer('bn2', dagnn.BatchNorm('numChannels', 64), {'conv_x2'}, {'bn_x2'}, {'bn2f', 'bn2b', 'bn2m'});

relu2 = dagnn.ReLU();
net.addLayer('relu2', relu2, {'bn_x2'}, {'relu_x2'}, {});

% ----------------------------------
% Stage 1: pooling
% ----------------------------------
% pool1 = dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]);
% net.addLayer('pool1', pool1, {'relu_x2'}, {'pool_x1','pool_x1_indices', 'sizes_pre_pool_x1', 'sizes_post_pool_x1'}, {});
pool1 = dagnn.Pooling('method', 'max', 'poolSize', [2 2], 'stride', 2);
net.addLayer('pool1', pool1, {'relu_x2'}, {'pool_x1'}, {});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                  STAGE II
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ----------------------------------
% Stage 2: 1st conv block
% ----------------------------------
conv3 = dagnn.Conv_original('size',[3,3,64,128], 'pad', 1, 'stride', 1, 'hasBias', true);
net.addLayer('conv3', conv3, {'pool_x1'},{'conv_x3'},{'conv_f3','conv_b3'});

net.addLayer('bn3', dagnn.BatchNorm('numChannels', 128), {'conv_x3'}, {'bn_x3'}, {'bn3f', 'bn3b', 'bn3m'});

relu3 = dagnn.ReLU();
net.addLayer('relu3', relu3, {'bn_x3'}, {'relu_x3'}, {});

% ----------------------------------
% Stage 2: 2nd conv block
% ----------------------------------
conv4 = dagnn.Conv_original('size',[3,3,128,128], 'pad', 1, 'stride', 1, 'hasBias', true);
net.addLayer('conv4', conv4, {'relu_x3'},{'conv_x4'},{'conv_f4','conv_b4'});

net.addLayer('bn4', dagnn.BatchNorm('numChannels', 128), {'conv_x4'}, {'bn_x4'}, {'bn4f', 'bn4b', 'bn4m'});

relu4 = dagnn.ReLU();
net.addLayer('relu4', relu4, {'bn_x4'}, {'relu_x4'}, {});

% ----------------------------------
% Stage 2: pooling
% ----------------------------------
% pool2 = dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]);
% net.addLayer('pool2', pool2, {'relu_x4'}, {'pool_x2','pool_x2_indices', 'sizes_pre_pool_x2', 'sizes_post_pool_x2'}, {});
pool2 = dagnn.Pooling('method', 'max', 'poolSize', [2 2], 'stride', 2);
net.addLayer('pool2', pool2, {'relu_x4'}, {'pool_x2'}, {});


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                  STAGE III
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ----------------------------------
% Stage 3: 1st conv block
% ----------------------------------
conv5 = dagnn.Conv_original('size',[3,3,128,256], 'pad', 1, 'stride', 1, 'hasBias', true);
net.addLayer('conv5', conv5, {'pool_x2'},{'conv_x5'},{'conv_f5','conv_b5'});

net.addLayer('bn5', dagnn.BatchNorm('numChannels', 256), {'conv_x5'}, {'bn_x5'}, {'bn5f', 'bn5b', 'bn5m'});

relu5 = dagnn.ReLU();
net.addLayer('relu5', relu5, {'bn_x5'}, {'relu_x5'}, {});

% ----------------------------------
% Stage 3: 2nd conv block
% ----------------------------------
conv6 = dagnn.Conv_original('size',[3,3,256,256], 'pad', 1, 'stride', 1, 'hasBias', true);
net.addLayer('conv6', conv6, {'relu_x5'},{'conv_x6'},{'conv_f6','conv_b6'});

net.addLayer('bn6', dagnn.BatchNorm('numChannels', 256), {'conv_x6'}, {'bn_x6'}, {'bn6f', 'bn6b', 'bn6m'});

relu6 = dagnn.ReLU();
net.addLayer('relu6', relu6, {'bn_x6'}, {'relu_x6'}, {});

% ----------------------------------
% Stage 3: pooling
% ----------------------------------
% pool3 = dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]);
% net.addLayer('pool3', pool3, {'relu_x6'}, {'pool_x3','pool_x3_indices', 'sizes_pre_pool_x3', 'sizes_post_pool_x3'}, {});
pool3 = dagnn.Pooling('method', 'max', 'poolSize', [2 2], 'stride', 2);
net.addLayer('pool3', pool3, {'relu_x6'}, {'pool_x3'}, {});


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                  STAGE IV
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ----------------------------------
% Stage 4: 1st conv block
% ----------------------------------
conv7 = dagnn.Conv_original('size',[3,3,256,512], 'pad', 1, 'stride', 1, 'hasBias', true);
net.addLayer('conv7', conv7, {'pool_x3'},{'conv_x7'},{'conv_f7','conv_b7'});

net.addLayer('bn7', dagnn.BatchNorm('numChannels', 512), {'conv_x7'}, {'bn_x7'}, {'bn7f', 'bn7b', 'bn7m'});

relu7 = dagnn.ReLU();
net.addLayer('relu7', relu7, {'bn_x7'}, {'relu_x7'}, {});

% ----------------------------------
% Stage 4: 2nd conv block
% ----------------------------------
conv8 = dagnn.Conv_original('size',[3,3,512,512], 'pad', 1, 'stride', 1, 'hasBias', true);
net.addLayer('conv8', conv8, {'relu_x7'},{'conv_x8'},{'conv_f8','conv_b8'});

net.addLayer('bn8', dagnn.BatchNorm('numChannels', 512), {'conv_x8'}, {'bn_x8'}, {'bn8f', 'bn8b', 'bn8m'});

relu8 = dagnn.ReLU();
net.addLayer('relu8', relu8, {'bn_x8'}, {'relu_x8'}, {});

% ----------------------------------
% Stage 4: pooling
% ----------------------------------
% pool4 = dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]);
% net.addLayer('pool4', pool4, {'relu_x8'}, {'pool_x4','pool_x4_indices', 'sizes_pre_pool_x4', 'sizes_post_pool_x4'}, {});
pool4 = dagnn.Pooling('method', 'max', 'poolSize', [2 2], 'stride', 2);
net.addLayer('pool4', pool4, {'relu_x8'}, {'pool_x4'}, {});


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                  STAGE V
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ----------------------------------
% Stage 3: 1st conv block
% ----------------------------------
conv9 = dagnn.Conv_original('size',[3,3,512,1024], 'pad', 1, 'stride', 1, 'hasBias', true);
net.addLayer('conv9', conv9, {'pool_x4'},{'conv_x9'},{'conv_f9','conv_b9'});

net.addLayer('bn9', dagnn.BatchNorm('numChannels', 1024), {'conv_x9'}, {'bn_x9'}, {'bn9f', 'bn9b', 'bn9m'});

relu9 = dagnn.ReLU();
net.addLayer('relu9', relu9, {'bn_x9'}, {'relu_x9'}, {});

% ----------------------------------
% Stage 3: 2nd conv block
% ----------------------------------
conv10 = dagnn.Conv_original('size',[3,3,1024,512], 'pad', 1, 'stride', 1, 'hasBias', true);
net.addLayer('conv10', conv10, {'relu_x9'},{'conv_x10'},{'conv_f10','conv_b10'});

net.addLayer('bn10', dagnn.BatchNorm('numChannels', 512), {'conv_x10'}, {'bn_x10'}, {'bn10f', 'bn10b', 'bn10m'});

relu10 = dagnn.ReLU();
net.addLayer('relu10', relu10, {'bn_x10'}, {'relu_x10'}, {});

% ----------------------------------
% Stage 3: unpooling
% ----------------------------------
% unpool1 = dagnn.Unpooling();
% net.addLayer('unpool1', unpool1, {'relu_x10', 'pool_x4_indices', 'sizes_pre_pool_x4', 'sizes_post_pool_x4'}, {'unpool_x1'}, {});
Upsample1=dagnn.ConvTranspose('size',[3,3,512,512],'hasBias',false,'upsample',[2,2],'crop',[0,1,0,1]);
net.addLayer('unpool1', Upsample1,{'relu_x10'},{'unpool_x1'},{'f1'});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                              UPCONV STAGE IV
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ----------------------------------
% Stage 4: concat block
% ----------------------------------
concat1 = dagnn.Concat('dim', 3);
net.addLayer('concat1', concat1, {'relu_x8', 'unpool_x1'}, {'concat_x1'}, {});

% ----------------------------------
% Stage 4: 1st conv block
% ----------------------------------
conv11 = dagnn.Conv_original('size',[3,3,1024,512], 'pad', 1, 'stride', 1, 'hasBias', true);
net.addLayer('conv11', conv11, {'concat_x1'}, {'conv_x11'}, {'conv_f11','conv_b11'});

net.addLayer('bn11', dagnn.BatchNorm('numChannels', 512), {'conv_x11'}, {'bn_x11'}, {'bn11f', 'bn11b', 'bn11m'});

relu11 = dagnn.ReLU();
net.addLayer('relu11', relu11, {'bn_x11'}, {'relu_x11'}, {});

% ----------------------------------
% Stage 4: 2nd conv block
% ----------------------------------
conv12 = dagnn.Conv_original('size',[3,3,512,256], 'pad', 1, 'stride', 1, 'hasBias', true);
net.addLayer('conv12', conv12, {'relu_x11'}, {'conv_x12'},{'conv_f12','conv_b12'});

net.addLayer('bn12', dagnn.BatchNorm('numChannels', 256), {'conv_x12'}, {'bn_x12'}, {'bn12f', 'bn12b', 'bn12m'});

relu12 = dagnn.ReLU();
net.addLayer('relu12', relu12, {'bn_x12'}, {'relu_x12'}, {});

% ----------------------------------
% Stage 4: unpooling
% ----------------------------------
% unpool2 = dagnn.Unpooling();
% net.addLayer('unpool2', unpool2, {'relu_x12', 'pool_x3_indices', 'sizes_pre_pool_x3', 'sizes_post_pool_x3'}, {'unpool_x2'}, {});
Upsample2=dagnn.ConvTranspose('size',[3,3,256,256],'hasBias',false,'upsample',[2,2],'crop',[1,0,1,0]);
net.addLayer('unpool2', Upsample2,{'relu_x12'},{'unpool_x2'},{'f2'});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                              UPCONV STAGE III
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ----------------------------------
% Stage 3: concat block
% ----------------------------------
concat2 = dagnn.Concat('dim', 3);
net.addLayer('concat2', concat2, {'relu_x6', 'unpool_x2'}, {'concat_x2'}, {});

% ----------------------------------
% Stage 3: 1st conv block
% ----------------------------------
conv13 = dagnn.Conv_original('size',[3,3,512,256], 'pad', 1, 'stride', 1, 'hasBias', true);
net.addLayer('conv13', conv13, {'concat_x2'}, {'conv_x13'}, {'conv_f13','conv_b13'});

net.addLayer('bn13', dagnn.BatchNorm('numChannels', 256), {'conv_x13'}, {'bn_x13'}, {'bn13f', 'bn13b', 'bn13m'});

relu13 = dagnn.ReLU();
net.addLayer('relu13', relu13, {'bn_x13'}, {'relu_x13'}, {});

% ----------------------------------
% Stage 3: 2nd conv block
% ----------------------------------
conv14 = dagnn.Conv_original('size',[3,3,256,128], 'pad', 1, 'stride', 1, 'hasBias', true);
net.addLayer('conv14', conv14, {'relu_x13'}, {'conv_x14'},{'conv_f14','conv_b14'});

net.addLayer('bn14', dagnn.BatchNorm('numChannels', 128), {'conv_x14'}, {'bn_x14'}, {'bn14f', 'bn14b', 'bn14m'});

relu14 = dagnn.ReLU();
net.addLayer('relu14', relu14, {'bn_x14'}, {'relu_x14'}, {});

% ----------------------------------
% Stage 3: unpooling
% ----------------------------------
% unpool3 = dagnn.Unpooling();
% net.addLayer('unpool3', unpool3, {'relu_x14', 'pool_x2_indices', 'sizes_pre_pool_x2', 'sizes_post_pool_x2'}, {'unpool_x3'}, {});
Upsample3=dagnn.ConvTranspose('size',[3,3,128,128],'hasBias',false,'upsample',[2,2],'crop',[0,1,0,1]);
net.addLayer('unpool3', Upsample3,{'relu_x14'},{'unpool_x3'},{'f3'});


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                              UPCONV STAGE II
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ----------------------------------
% Stage 2: concat block
% ----------------------------------
concat3 = dagnn.Concat('dim', 3);
net.addLayer('concat3', concat3, {'relu_x4', 'unpool_x3'}, {'concat_x3'}, {});

% ----------------------------------
% Stage 2: 1st conv block
% ----------------------------------
conv15 = dagnn.Conv_original('size',[3,3,256,128], 'pad', 1, 'stride', 1, 'hasBias', true);
net.addLayer('conv15', conv15, {'concat_x3'}, {'conv_x15'}, {'conv_f15','conv_b15'});

net.addLayer('bn15', dagnn.BatchNorm('numChannels', 128), {'conv_x15'}, {'bn_x15'}, {'bn15f', 'bn15b', 'bn15m'});

relu15 = dagnn.ReLU();
net.addLayer('relu15', relu15, {'bn_x15'}, {'relu_x15'}, {});

% ----------------------------------
% Stage 2: 2nd conv block
% ----------------------------------
conv16 = dagnn.Conv_original('size',[3,3,128,64], 'pad', 1, 'stride', 1, 'hasBias', true);
net.addLayer('conv16', conv16, {'relu_x15'}, {'conv_x16'},{'conv_f16','conv_b16'});

net.addLayer('bn16', dagnn.BatchNorm('numChannels', 64), {'conv_x16'}, {'bn_x16'}, {'bn16f', 'bn16b', 'bn16m'});

relu16 = dagnn.ReLU();
net.addLayer('relu16', relu16, {'bn_x16'}, {'relu_x16'}, {});

% ----------------------------------
% Stage 2: unpooling
% ----------------------------------
% unpool4 = dagnn.Unpooling();
% net.addLayer('unpool4', unpool4, {'relu_x16', 'pool_x1_indices', 'sizes_pre_pool_x1', 'sizes_post_pool_x1'}, {'unpool_x4'}, {});
Upsample4=dagnn.ConvTranspose('size',[3,3,64,64],'hasBias',false,'upsample',[2,2],'crop',[0,1,0,1]);
net.addLayer('unpool4', Upsample4,{'relu_x16'},{'unpool_x4'},{'f4'});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                              UPCONV STAGE I
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ----------------------------------
% Stage 1: concat block
% ----------------------------------
concat4 = dagnn.Concat('dim', 3);
net.addLayer('concat4', concat4, {'relu_x2', 'unpool_x4'}, {'concat_x4'}, {});

% ----------------------------------
% Stage 1: 1st conv block
% ----------------------------------
conv17 = dagnn.Conv_original('size',[3,3,128,64], 'pad', 1, 'stride', 1, 'hasBias', true);
net.addLayer('conv17', conv17, {'concat_x4'}, {'conv_x17'}, {'conv_f17','conv_b17'});

net.addLayer('bn17', dagnn.BatchNorm('numChannels', 64), {'conv_x17'}, {'bn_x17'}, {'bn17f', 'bn17b', 'bn17m'});

relu17 = dagnn.ReLU();
net.addLayer('relu17', relu17, {'bn_x17'}, {'relu_x17'}, {});

% ----------------------------------
% Stage 1: 2nd conv block
% ----------------------------------
conv18 = dagnn.Conv_original('size',[3,3,64,64], 'pad', 1, 'stride', 1, 'hasBias', true);
net.addLayer('conv18', conv18, {'relu_x17'}, {'conv_x18'},{'conv_f18','conv_b18'});

net.addLayer('bn18', dagnn.BatchNorm('numChannels', 64), {'conv_x18'}, {'bn_x18'}, {'bn18f', 'bn18b', 'bn18m'});

relu18 = dagnn.ReLU();
net.addLayer('relu18', relu18, {'bn_x18'}, {'relu_x18'}, {});

% ----------------------------------
% Stage 0: Prediction block
% ----------------------------------
pred = dagnn.Conv_original('size',[1,1,64,1], 'pad', 0, 'stride', 1, 'hasBias', true);
net.addLayer('pred', pred, {'relu_x18'},{'Image_Pre'},{'pred_f1','pred_b1'});
SumBlock=dagnn.Sum();
net.addLayer('sum',SumBlock,{'Image_Pre','FBP'},{'Image'});


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Projection net%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                  LOSS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SmoothBlock=dagnn.Smooth();
net.addLayer('Smooth', SmoothBlock, {'Image'}, {'loss'}) ;
PrjCompareBlock=dagnn.PrjCompare();
net.addLayer('PrjCompare', PrjCompareBlock, {'Image','data','Hsys','weights'}, {'loss2'}) ;
net.initParams() ;
for i=1:numel(net.vars)
    net.vars(i).precious=0;
end
net.vars(end-5).precious=1;
net.vars(end-4).precious=1;
net.vars(end).precious=1;


% load 'net.mat'
% for i=1:numel(net2.params)
%  net.params(i).value=net2.params(i).value;
% end


