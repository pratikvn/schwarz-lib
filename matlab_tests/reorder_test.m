clear all

A = delsq(numgrid('S',20));
N = length(A);

r = symamd(A);
for i=1:length(r)
    ir(r(1,i)) = i;
end
L1 = chol(A);
L2 = chol(A(r,r));
subplot(1,2,1),spy(L1),title('L')
subplot(1,2,2),spy(L2),title('L(p,p)')
L1t = transpose(L1);
L2t = transpose(L2);

b = rand(size(A,1),1);
br = b(r,1);
err1 = norm(abs(b-br),2);
x_it = pcg(A,b,1e-10,1000);
x_di = (L1\(L1t\b));
x_di2 = (L2\(L2t\br));
x_di3 = A(r,r)\br;
x_di2 = x_di2(ir,1);
x_di3 = x_di3(ir,1);

err = norm(abs(x_di-x_it),2);
err2 = norm(abs(x_di2-x_it),2);
err3 = norm(abs(x_di3-x_it),2);



