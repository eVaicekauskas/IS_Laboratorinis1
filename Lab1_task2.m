clc
close all
clear all

%Classification using naive Bayes 

% given data:
% x1[] - first feature array, x2[] - second feature array, T - expected outcome
x1 = [0.21835 0.14115 0.37022 0.31565 0.36484 0.46111 0.55223 ...
    0.16975 0.49187 0.14913 0.18474 0.08838 0.098166];
x2 = [0.81884 0.83535 0.8111 0.83101 0.8518 0.82518 0.83449 ...
    0.84049 0.80889 0.77104 0.6279 0.62068 0.79092];
Yinitial = [1 1 1 1 1 1 1 1 1 0 0 0 0];

plot(x1(1:9),x2(1:9),'ro')
hold on
plot(x1(10:13),x2(10:13),'go')


% feature x1 binarization (colour red - 1, colour green - 0)
% feature x2 binarization (shape round - 1, shape not-round - 0)
for i = 1:length(x1)
    
    %threshold is based on the inspection of the x1 data
    if x1(i) >= 0.1450 
        A(i) = 1; 
    else
        A(i) = 0;
    end
    
    %threshold is based on the inspection of the x2 data
    if x2(i) >= 0.8 
        B(i) = 1; 
    else
        B(i) = 0;
    end
    
end

plot(0.1450,0.6:0.001:0.95,'k.')
plot(0:0.001:0.6,0.8,'b.')
title('Duomenų rinkinys ir du slenksčiai parametrams')
xlabel('x1 - spalva')
ylabel('x2 - apvalumas')

%% single case example

% case 3 - A = 1, B = 1

%P(Y=0|A=1,B=1) -> P(Y=0,A=1,B=1) = P(Y=0)P(A=1|Y=0)P(B=1|Y=0) =
%=(4/13)(2/4)(1/4) = 8/208 = 1/26 ~ 0.0385

%P(Y=1|A=1,B=1) -> P(Y=1,A=1,B=1) = P(Y=1)P(A=1|Y=1)P(B=1|Y=1) =
%=(9/13)(8/9)(9/9) = 648/1053 = 8/13 ~ 0.6154

%P(Y=0|A=1,B=1) < P(Y=1|A=1,B=1) = 1/26 <  8/13,
%therefore case 3 is labeled as 1 (Yi = 1) 

%% classification 

%determining the probabilities
selectionArray = [1 2 3 4 11 12 13]; % data points used for determining the probabilities:
% P(Y),P(A|Y),P(B|Y)

%finding P(Y) for a given set
Y = [];
for i = 1:length(selectionArray)
    id = selectionArray(i);
    Y(i) = Yinitial(id);
end
PY(1) = 1-sum(Y)/length(Y); %P(Y=0)
PY(2) = sum(Y)/length(Y); %P(Y=1)

instanceCount = [0 0 0 0 0 0 0 0];

PAY = [0 0 0 0];
PBY = [0 0 0 0];

%finding P(A|Y), P(B|Y) for a given set 
for i = 1:length(selectionArray)
    id = selectionArray(i);
    
    if A(id)==0 && Yinitial(id)==0
        instanceCount(1) = instanceCount(1) + 1;
        PAY(1) = instanceCount(1)/(PY(1)*length(Y)); %P(A=0|Y=0)
    elseif  A(id)==1 && Yinitial(id)==0
       instanceCount(2) = instanceCount(2) + 1; 
       PAY(2) = instanceCount(2)/(PY(1)*length(Y)); %P(A=1|Y=0)
    elseif A(id)==0 && Yinitial(id)==1
        instanceCount(3) = instanceCount(3) + 1; 
        PAY(3) = instanceCount(3)/(PY(2)*length(Y)); %P(A=0|Y=1)
    elseif  A(id)==1 && Yinitial(id)==1
       instanceCount(4) = instanceCount(4) + 1; 
       PAY(4) = instanceCount(4)/(PY(2)*length(Y)); %P(A=1|Y=1)
    end
    
    if B(id)==0 && Yinitial(id)==0
        instanceCount(5) = instanceCount(5) + 1;
        PBY(1) = instanceCount(5)/(PY(1)*length(Y)); %P(B=0|Y=0)
    elseif  B(id)==1 && Yinitial(id)==0
       instanceCount(6) = instanceCount(6) + 1; 
       PBY(2) = instanceCount(6)/(PY(1)*length(Y)); %P(B=1|Y=0)
    elseif B(id)==0 && Yinitial(id)==1
        instanceCount(7) = instanceCount(7) + 1; 
        PBY(3) = instanceCount(7)/(PY(2)*length(Y)); %P(B=0|Y=1)
    elseif  B(id)==1 && Yinitial(id)==1
       instanceCount(8) = instanceCount(8) + 1; 
       PBY(4) = instanceCount(8)/(PY(2)*length(Y)); %P(B=1|Y=1)
    end

end
    
selectionArray = [5 6 10]; % out of the given data points, this set will be classified  
for i = 1:length(selectionArray)
    id = selectionArray(i);

    if A(id) == 1
        pay = PAY(2); %P(A=1|Y=0)
    else 
        pay = PAY(1); %P(A=0|Y=0)
    end
    
    if B(id) == 1
        pby = PBY(2); %P(B=1|Y=0)
    else 
        pby = PBY(1); %P(B=0|Y=0)
    end
    P1 = PY(1)*pay*pby; %P(Y=0|A,B)
    
    if A(id) == 1
        pay = PAY(4); %P(A=1|Y=1)
    else 
        pay = PAY(3); %P(A=0|Y=1)
    end
    
    if B(id) == 1
        pby = PBY(4); %P(B=1|Y=1)
    else 
        pby = PBY(3); %P(B=0|Y=1)
    end
    P2 = PY(2)*pay*pby; %P(Y=1|A,B)
    
    
    if P1 > P2
        label(i) = 0;
    else
        label(i) = 1;
    end
    
    if label(i) == Yinitial(id)
        error(i) = 0;
    else 
        error(i) = 1;
    end
    
    
end

error

%% Išvados
% Šis Naive Bayes klasifikavimo pavyzdys veikia, kol, pavyzdžiui - mokomajame rinkinyje raudonų kriaušių (obuolio sąvybė) yra mažiau nei ne raudonų kriaušių. ...
... šiuo atveju mokomasis rinkinys mažas todėl su pasirinktais slenksčiais pusė kriaušių yra apvalios. Taigi, klasifikavimo sėkmingumas priklauso nuo mokymo ...
... imties pasirinkimo. Didesniame rinkinyje ši priklausomybė nebūtų tokia svarbi, nes parametro vidurkis būtų tikslesnis, leidžiant parinkti slenkstį tiksliau. 

