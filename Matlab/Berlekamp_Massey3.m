% Sacado de aquí
%http://home.ie.cuhk.edu.hk/~wkshum/wordpress/?p=1414

% Berlekamp-Massey algorithm for finding the shortest
% linear feed-back shift register, aka the linear complexity,
% of a binary sequence.
%
% Input s is a vector whose components are either 0 or 1.
% s represents a binary sequence of length length(s).
% Output f represents the shortest feedback polynomial p(x) of sequence s
% The i-th component in f is the coefficient of x^(i-1) in p(x).
% The number of components in f equals to the linear complexity plus one.
% The first entry in f must be equal to 1, 
% but the last entry in f may equal 0.
%
% The returned value f satisfies the property that
% the entries in mod(conv(f,s),2) are all zeros except the first and last
% length(f)-1 entries. 
%
% Output LCP is the linear complexity profile
% The i-th entry of LCP is the linear complexity of the first 
% i bits of the binary sequence s.
function [f_cell, LCP] = Berlekamp_Massey3(s)
LCP = zeros(1,length(s)); % Linear complexity profile
f = [1]; % feedback polynomial initialized to the constant polynomial 1
g = [1]; % initialized to the constant polynomial 1
m = 0;
for n = 1:length(s)
 if mod(n,10000)==0
     disp(n)
 end    
 if mod(f * s(n:-1:(n-length(f)+1))',2) == 1 % check if discrepancy is not zero
 % Compute the shortest feedback polynomial for the first n bits of s
   L = max(length(f), n-length(f)+2);
   new = mod([f zeros(1,L-length(f))] + [zeros(1,n-m) g zeros(1,L-n+m-length(g))],2);
   if L > length(f) % if the degree strictly increase
     g = f; % record the feedback polynomial
     m = n; % and when the increase occurs
   end
   f = new;
 end
 LCP(n) = length(f)-1;
 f_cell{n}=f;
end