%% create a row vector
a = [1 2 3 4]

%% create a column vector
a = [1; 2; 3; 4]

%% vector access
a(1),  a(2)     %index starts from 1

a(end)          %access last element

a(2:3),  a(2:end)   %access a subsequence

%% create a matrix
A = [1 2 3; 4 5 6; 7 8 9]

%% indexing
A(2,3)
A(2)
A(4)            %column-major indexing; matrix stored in a 1 dimensional array
                %always scan a matrix in columns instead of rows

A(2:3,1:2)      

A(:)            %retrieve all elements

A(1,:)          %retrieve the first row

A(:,2)          %retrive the first column

A([1 2], [1 3])     %retrive row 1 & 2 and column 1 & 3

%% transpose
A'

%% operators: +, - *, /  : all in matrix algebra
B = rand(2,3);
C = rand(3,4);

%%
B + rand(2,3)

%%
B + rand(2,4)   %Error: Matrix dimensions must agree

B - rand(2,4)   %Error: Matrix dimensions must agree

%%
B*C             %matrix multiplication

%%
C*B             %Error: Inner matrix dimensions must agree

%%
B/C             %solves for x*C = B;  B and C must have the same number of columns; rarely used

%% the mighty backslash operator
B\C             %solves for B*x = C;  B and C must have the same number of rows; 
                %C is often a vector; B usually has more rows than columns

%%element-wise operation: .+, .-, .*, ./, .^
B = rand(2,3);
C = rand(2,3);

%%
B.*C

B./C

B.^2            %element-wise square


%% inner product of two vectors
a = rand(10,1);
b = rand(10,1);

ans = 0;
for i = 1:10
    ans = a(i)*b(i);
end
disp(ans);
%WARNING: always try to avoid for-loop or while-loop if possible!!!

%% alternative
sum(a.*b);

%% or in matrix multiplication
a'*b                

%% or use matlab builtin function
dot(a,b)


%% work with images and the Matlab Image Processing Toolbox
I = imread('einstain.png');
figure; imshow(I);

size(I)

class(I)            
I2 = im2double(I);  %data type transform
class(I2)
max(I2(:))

I3 = im2uint8(I2);
I4 = im2uint16(I2);
class(I3)
class(I4)

figure; imshow(I2);
figure; imshow(I3);
figure; imshow(I4);

figure; imagesc(I);     %display image with scaled colors
colormap(gray);
colormap(jet);


%% color images
I = imread('woman.png');
size(I)

I2 = rgb2gray(I);
I3 = uint8(mean(I, 3));  %what's the difference?

d = abs(I2-I3);
figure; imagesc(d);


%% resize
I4 = imresize(I, 0.5);
I5 = imresize(I4, 8, 'bilinear');
I6 = imresize(I4, 8, 'nearest');

imwrite(I2, 'woman_gray.png');
imwrite(I5, 'woman_resize_bilinear.bmp');
imwrite(I6, 'woman_resize_nearest.jpg');

%how about this?
I7 = I(1:2:end, 1:2:end,:);          
I8 = I(1:8:end, 1:8:end,:);         %correct way: prefiltering and downsample; avoid aliasing


%% image crop
figure; imshow(I);
[x y] = ginput(2);

I_crop = I(y(1):y(2), x(1):x(2),:);
figure; imshow(I_crop);

rect = getrect;
I_crop2 = imcrop(I, rect);
figure; imshow(I_crop2);


%% compute gradient
I = im2double(imread('einstain.png'));

dx = I(:, 2:end) - I(:, 1:end-1);
figure; 
imagesc(abs(dx), [0, max(abs(dx(:)))]);
axis image;

dy = I(2:end,:) - I(1:end-1,:);
figure;
imagesc(abs(dy), [0, max(abs(dy(:)))]);
axis image;

dx = [dx zeros(size(I,1),1)];
dy = [dy; zeros(1, size(I,2))];
grad = sqrt(dx.*dx + dy.*dy);
figure; imagesc(grad, [min(grad(:)) max(grad(:))]);
axis image;

%% edge detection
edge_s = edge(I, 'sobel');
edge_c = edge(I, 'canny');





%% extra exercise: find the closest number for each element of b in a
a = [51   47   53    2   21   39   57   20   31    7];
b = [56   75   13   30   35    8   30   28   90   93];










a2 = repmat(a, [10, 1]);
b2 = repmat(b', [1, 10]);
dist = abs(a2 - b2);            %do the subtraction calculation in batch!
[~, idx] = min(dist, [], 2);    %use matlab builtin function to do the search
disp([b; a(idx)])



%%a simplified version of patch match with a not all neighbors synthesized
input = rand(100,100);
p = rand(3,3);                      %3x3 window size; not the 3x3 seed we talked about in class. Pixel in the middle to be synthesized
filledmap = [1 1 1; 1 0 0; 1 0 0];  %a binary map indicating which pixels are synthesized in the patch

candidates = im2col(input, [3 3], 'sliding');
p_vec = reshape(p, [], 1);
p_vec2 = p_vec.*reshape(filledmap, [],1);

p_vec2_rep = repmat(p_vec2, 1, size(candidates,2));
dist = sum((candidates - p_vec2_rep).^2);               %in real implementation, add gaussian weights
[~, idx] = min(dist);
[matchedrow, matchedcol]= ind2sub([100-3+1 100-3+1], idx);  %figure out what this line is doing by yourself..






