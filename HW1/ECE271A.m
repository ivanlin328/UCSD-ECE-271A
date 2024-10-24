clear all
clc

load('homework1/TrainingSamplesDCT_8.mat')
foreground = TrainsampleDCT_FG;  
background = TrainsampleDCT_BG;  
[m,n] =size(TrainsampleDCT_FG);  %% 250*64
[a,b] =size(TrainsampleDCT_BG);  %% 1053*64
zig=load('homework1/Zig-Zag Pattern.txt');
zig=zig+1;
cheetah_image =im2double(imread('homework1/cheetah.bmp'));

%%Training
%% Calculate the prior pobability of cheetah and grace,P(cheetah)and P(grass)
total = m+a;
prior_cheetah = m / total;
prior_grass = a / total;

%% Find the index of 2nd largest coefficient in each row 
FG=[];
BG=[];
for i=1:m
   [~, sort_row] = sort(foreground(i,:));
   FG=[FG,sort_row(end-1)];
end
%%disp(sort_row)
%%disp(FG)
for j=1:a
   [~, sort_row2] = sort(background(j,:));
   BG=[BG,sort_row2(end-1)];
end
%%disp(BG)
%% show the result on the histogram
figure;  
FG_histogram=histogram(FG','BinEdges',1:65); % Plot the histogram 
title('P_{X|Y}(X|cheetah))');
xlabel('Index');
ylabel('Frequency');
xticks(1:64);

figure;  
BG_histogram=histogram(BG,'BinEdges',1:65);  % Plot the histogram
title('P_{X|Y}(X|background)');
xlabel('Index');
ylabel('Frequency');
xticks(1:64);

P_X_cheetah=FG_histogram.Values/size(foreground,1);  %%sum of each index in the histogram / 250
P_X_background= BG_histogram.Values/size(background,1);   %%sum of each index histogram / 1053

%% Classification
A_matrix=[];
[row,column]=size(cheetah_image); %%255*270
block_size = 8;
for i = 1 : row-8
    for j= 1 :column-8
        block=cheetah_image(i:i+7, j:j+7);
        dct_block=dct2(block);
        dct_block=dct_block(:);
        zig=zig(:);
        v(zig)= abs(dct_block);
        [value,index] = sort(v,'descend');
        %%choose the second largest coefficient and apply BDF
        A_matrix(i,j)=prior_cheetah*P_X_cheetah(index(1,2))>=prior_grass*P_X_background(index(1,2));  
    end
end

A_matrix=reshape(A_matrix,247,262);
padd_cheetah = padarray(A_matrix,[4,4],'both');
%disp(size(padd_cheetah)) %255*270
%% show the result
figure;
colormap(gray(255)) 
imagesc(padd_cheetah)

%%calculate the error
cheetah_mask =imread('homework1/cheetah_mask.bmp');

cheetah_mask_one=find(cheetah_mask); %%index which value is 0
cheetah_mask_wrong_1=sum(padd_cheetah(cheetah_mask_one)==0);

cheetah_mask_zero=find(~cheetah_mask);%%index which value is 1
cheetah_mask_wrong_0=sum(padd_cheetah(cheetah_mask_zero)==1);

size_of_pic=length(cheetah_mask_one)+length(cheetah_mask_zero);

p_error = (cheetah_mask_wrong_1+cheetah_mask_wrong_0)/size_of_pic
 











