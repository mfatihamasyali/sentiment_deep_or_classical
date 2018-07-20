function MPadded = leftPad(M,N)

[dimension,sequenceLength] = size(M);
paddingLength = N - sequenceLength;
MPadded = [zeros(dimension,paddingLength) M];

end