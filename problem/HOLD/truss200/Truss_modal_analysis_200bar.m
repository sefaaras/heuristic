function [OUTPUT]=Truss_modal_analysis_200bar(Individual)

Truss_Data=TrussData_modal_200bar;

Truss_Data.A(1:4,:)=Individual(1);
Truss_Data.A([5,8,11,14,17],:)=Individual(2);
Truss_Data.A([19,20,21,22,23,24],:)=Individual(3);
Truss_Data.A([18,25,56,63,94,101,132,139,170,177],:)=Individual(4);
Truss_Data.A([26,29,32,35,38],:)=Individual(5);
Truss_Data.A([6,7,9,10,12,13,15,16,27,28,30,31,33,34,36,37],:)=Individual(6);
Truss_Data.A([39,40,41,42],:)=Individual(7);
Truss_Data.A([43,46,49,52,55],:)=Individual(8);
Truss_Data.A([57,58,59,60,61,62],:)=Individual(9);
Truss_Data.A([64,67,70,73,76],:)=Individual(10);
Truss_Data.A([44,45,47,48,50,51,53,54,65,66,68,69,71,72,74,75],:)=Individual(11);
Truss_Data.A([77,78,79,80],:)=Individual(12);
Truss_Data.A([81,84,87,90,93],:)=Individual(13);
Truss_Data.A([95,96,97,98,99,100],:)=Individual(14);
Truss_Data.A([102,105,108,111,114],:)=Individual(15);
Truss_Data.A([82,83,85,86,88,89,91,92,103,104,106,107,109,110,112,113],:)=Individual(16);
Truss_Data.A([115,116,117,118],:)=Individual(17);
Truss_Data.A([119,122,125,128,131],:)=Individual(18);
Truss_Data.A([133,134,135,136,137,138],:)=Individual(19);
Truss_Data.A([140,143,146,149,152],:)=Individual(20);
Truss_Data.A([120,121,123,124,126,127,129,130,141,142,144,145,147,148,150,151],:)=Individual(21);
Truss_Data.A([153,154,155,156],:)=Individual(22);
Truss_Data.A([157,160,163,166,169],:)=Individual(23);
Truss_Data.A([171,172,173,174,175,176],:)=Individual(24);
Truss_Data.A([178,181,184,187,190],:)=Individual(25);
Truss_Data.A([158,159,161,162,164,165,167,168,179,180,182,183,185,186,188,189],:)=Individual(26);
Truss_Data.A([191,192,193,194],:)=Individual(27);
Truss_Data.A([195,197,198,200],:)=Individual(28);
Truss_Data.A([196,199],:)=Individual(29);

Truss_Data.A=Truss_Data.A*10^(-4);

dnsay=size(Truss_Data.dnkoor,1);
elsay=size(Truss_Data.eldn,1);

topser=dnsay*3;

[rijitlik,elboy]=stiffness_truss(topser,elsay,Truss_Data.eldn,Truss_Data.dnkoor,Truss_Data.E,Truss_Data.A);
[mass_mat]=mass_truss(topser,elsay,Truss_Data.eldn,Truss_Data.dnkoor,Truss_Data.GS,Truss_Data.A);
for i=1:dnsay
mass_mat(3*i-2,3*i-2)=mass_mat(3*i-2,3*i-2)+Truss_Data.add_mass(i);
mass_mat(3*i-1,3*i-1)=mass_mat(3*i-1,3*i-1)+Truss_Data.add_mass(i);
mass_mat(3*i,3*i)=mass_mat(3*i,3*i)+Truss_Data.add_mass(i);
end
mass_mat=mass_mat.*(mass_mat>10^(-6));

tutser=find(Truss_Data.MesKos'==1);
ser=setdiff([1:topser]',[tutser]);
[FI,LAMDA]=eig(inv(mass_mat(ser,ser))*rijitlik(ser,ser));

[LAMDA_sorted, ind]=sort(diag(LAMDA),'ascend');

Frekans=round(sqrt(abs(LAMDA_sorted))/(2*pi),3);

%% CONSTRAINTS
for i=1:3
if Frekans(i)<Truss_Data.minfre(i)
    ck(i)=abs((Frekans(1)-Truss_Data.minfre(i))/Truss_Data.minfre(i));
else
    ck(i)=0;
end
end

%% OBJECT FUNCTION
ObjVal=0;
for i=1:size(Truss_Data.eldn,1)
ObjVal=ObjVal+elboy(i)*Truss_Data.A(i)*Truss_Data.GS;
end
OUTPUT.OBJVAL=ObjVal;

%% TOTAL CONSTRAINT VIOLATIONS
OUTPUT.VIO=sum(ck);

%% PENALISED OBJECTIVE FUNCTION
OUTPUT.PENALISED=ObjVal*(1+10^8*sum(ck));  % Penalty function

