function C = doc2sequence(emb,documents)

for i = 1:numel(documents)
    words = string(documents(i));
    idx = ~ismember(emb,words);
    words(idx) = [];
    C{i} = word2vec(emb,words)';
end

end