M=[4 5 6;2 1 4;1 0 5]
[min_val,idx]=min(M(:))
[row,col]=ind2sub(size(M),idx)