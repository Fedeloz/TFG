% takes a matrix, modifies or displace n_bits and
% saves the Berlekamp-Massey result matrix
%-----------------------------------------------------%
% Option 0 -> urand
% Option 1 -> parity
% Option 2 -> modify parity, reverse bits
% Option 3 -> modify parity, displace bits
%-----------------------------------------------------%
clear;
close all;
%-----------------------------------------------------%
option = 0;
n_bits = 0;     % [1, 5, 10, 20, 35, 50]
%tam=1*1e5;
tam=6*1e4;      % 1e5 is too much
if option == 0
    path = 'originales/urand1024_2.mat';    % R -> True random
    name = 'Mat_LCP_Urand2';
else
    path = 'originales/parity1024m.mat';    % L -> Pseudo random 
    name = 'Mat_LCP_Parity';
end
eval(sprintf('load(''%s'');',path));
original = Data(1:tam);
%clearvars -except original option n_bits name tam
%-----------------------------------------------------%
% modificamos array extraído de la matriz si se pide
%-----------------------------------------------------%
if option == 2  % Reverse n_bits
    for i = 1:n_bits
        original(1, i) = ~original(1, i);
    end
    name = [name '_' num2str(n_bits) '_reversed'];
elseif option == 3
    original = circshift(original, n_bits);
    name = [name '_' num2str(n_bits) '_displaced'];
end
%-----------------------------------------------------%
% aplicamos Berlekamp-Massey al vector
%-----------------------------------------------------%
% f_cell  -> shortest feedback polynomial p(x) of sequence s
% LCP_aux -> linear complexity profile
[f_cell LCP_aux]=Berlekamp_Massey3(original);
%-----------------------------------------------------%
% Nos quedamos con la matriz cuadrada izquierda
%-----------------------------------------------------%
f_cell_recortado=f_cell(round(tam*0.75)+1:end); % last 25% of f_cell
tam2=length(f_cell_recortado);

size_matriz = length(f_cell_recortado{end});
Pol_matriz = -1*ones(tam2,size_matriz);         % template matrix
for i=1:tam2
    Pol_matriz(i,1:length(f_cell_recortado{i}))=f_cell_recortado{i};
end 

Mat_coef_recortada=Pol_matriz(:,1:tam2);
%-----------------------------------------------------%
% save matrix
%-----------------------------------------------------%
name = ['matrix_generated/' name '.mat'];
save(name, 'Mat_coef_recortada', '-v7.3');
