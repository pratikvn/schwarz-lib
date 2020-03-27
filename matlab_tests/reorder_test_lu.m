  clear all
% 
A = delsq(numgrid('S',20));
N = length(A);

r = symamd(A);
for i=1:length(r)
    ir(r(1,i)) = i;
end
[L1,U1] = lu(A);
[L2,U2,p,q] = lu(A);
[L3,U3,p1] = lu(A,'vector');
% subplot(2,3,1),spy(L1),title('L')
% subplot(2,3,2),spy(L2),title('L(p,p)')
% subplot(2,3,3),spy(L3),title('L3(p,p)')
% subplot(2,3,4),spy(U1),title('U')
% subplot(2,3,5),spy(U2),title('U(p,p)')
% subplot(2,3,6),spy(U3),title('U3(p,p)')
L1t = transpose(L1);
L2t = transpose(L2);
fmat=L2*U2;



subplot(1,2,1),spy(p*p'),title('P')
subplot(1,2,2),spy(A),title('Q')

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

% Lmat1=load("L_mat_0.csv");
% Umat1=load("U_mat_0.csv");
% localmat1=load("local_mat_0.csv");
% 
% L=sparse(Lmat1(:,1),Lmat1(:,2),Lmat1(:,3));
% U=sparse(Umat1(:,1),Umat1(:,2),Umat1(:,3));
% mat=sparse(localmat1(:,1),localmat1(:,2),localmat1(:,3));
% 
% r=load("perm_0.csv");
% ur=load("umfperm_0.csv");
% qw=r-ur;
% r2=r;
% %r=perm1.VarName1+1;
% %ir=perm1.VarName1+1;
% r=r+1;
% ir=r;
% for i=1:length(r)
%     ir(r(i,1)) = i;
% end
% b = rand(size(mat,1),1);
% br = b(r,1);
% 
% fmat = L(ir,ir)*U(ir,ir)';
% % fmat = fmat;
% subplot(2,2,1),spy(L),title('L')
% subplot(2,2,2),spy(U),title('U')
% % subplot(2,2,3),spy(L(ir,ir)*U(ir,ir)),title('L*U')
% subplot(2,2,3),spy(fmat),title('L*U')
% subplot(2,2,4),spy(mat),title('A')
% Lt = transpose(L);
% U = transpose(U);
% x_it = pcg(mat,b,1e-10,3000);
% y=(L\b(r,1));
% x_di2 = U\y;
% x_di4 = (L\(Lt\br));
% x_di3 = mat(r,r)\br;
% x_di2 = x_di2(ir,1);
% x_di3 = x_di3(ir,1);
% x_di4 = x_di4(ir,1);
% 
% %err = norm(abs(x_di-x_it),2);
% err2 = norm(abs(x_di2-x_it),2);
% err3 = norm(abs(x_di3-x_it),2);
% err4 = norm(abs(x_di4-x_it),2);
% 
% for i=1:size(U,1) 
% diagU(i) = U(i,i);
% end
% for i=1:size(L,1) 
% diagL(i) = L(i,i);
% end
