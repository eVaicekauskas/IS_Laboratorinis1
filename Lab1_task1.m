clc
%close all
clear all

% CLASSIFICATION USING PERCEPTRON LAB1_TASK1

% given data:
% x1[] - first feature array, x2[] - second feature array, T - expected outcome
x1 = [0.21835 0.14115 0.37022 0.31565 0.36484 0.46111 0.55223 ...
    0.16975 0.49187 0.14913 0.18474 0.08838 0.098166];
x2 = [0.81884 0.83535 0.8111 0.83101 0.8518 0.82518 0.83449 ...
    0.84049 0.80889 0.77104 0.6279 0.62068 0.79092];
T = [1 1 1 1 1 1 1 1 1 -1 -1 -1 -1];


%% training single perceptron

n = 0.1; % learning step
Esum = 1; % sum of errors 
counter = 0; % counter for debuging
once = 1; % variable for enabling a one time execution of a part of the code
trainedAndValidated = 0; % value determining wether the classifier was also validated after training

while trainedAndValidated ~= 1 % loop is exited if the classifier is validated after training
    
    w1 = randn(1); % first weight
    w2 = randn(1); % second weight
    b = randn(1); % parameter
    
    selectionArray = [1 2 3 10 11]; % out of all samples 3 apples of id's 1, 2 and 3, and pears of id's 10, 11 are selected for training
    
    while Esum ~= 0 % executes while the total error is not 0
        
        counter  = counter +1;

        for i = 1 : length(selectionArray)

            id = selectionArray(i); % index of the original arrays x1[], x2[] and T[]
            v = x1(id)*w1 + x2(id)*w2 + b; 

            if v > 0
                y = 1;
            else
                y = -1;
            end
      
            e = T(id) - y; % current error
        
            if e~=0
               w1 = w1 + n*e*x1(id); % parameters are adjusted according to the error 
               w2 = w2 + n*e*x2(id);
               b = b + n*e*1;

            end

            E(i) = e; % array storing current example error 

            if once == 1 % ploting the training sample values
                if id <= 9
                    plot(x1(id),x2(id),'ro')
                    hold on 
                else
                    plot(x1(id),x2(id),'go')
                    hold on
                end
            end

        end
        once = 0;
        Esum = sum(abs(E)); 

    end
    % graphical verification (representation)
    X = 0:0.1:0.8;
    Y = -X*w1/w2 - b/w2;

    plot(X,Y)
    xlim([0,0.5])
    ylim([0.5,1])

    %% validation
    selectionArray = [4 5 6 12]; % different samples are selected for verification 
    for i = 1:length(selectionArray)
        id = selectionArray(i);
        v = x1(id)*w1 + x2(id)*w2 + b;

        if v > 0
            y = 1;
        else
            y = -1;
        end

        e = T(id) - y;
        E(i) = e;

        if id <= 9
            plot(x1(id),x2(id),'r*') 

        else
            plot(x1(id),x2(id),'g*')
        end

    end
    
    Esum = sum(abs(E));

    if Esum == 0 
        trainedAndValidated = 1;
    else 
        trainedAndValidated = 0;
    end
    
end

 %% testing 
 
 selectionArray = [7 8 9 13]; % different samples are selected for testing
 
for i = 1:length(selectionArray)
    
    id = selectionArray(i);
 
    v = x1(id)*w1 + x2(id)*w2 + b;

    if v > 0
        y = 1;
    else
        y = -1;
    end

    e = T(id) - y;
    E(i) = e;

    if id <= 9
        plot(x1(id),x2(id),'rx') 
    else
        plot(x1(id),x2(id),'gx')
    end

end

Esum = sum(abs(E));

if Esum == 0 
    fprintf('no mistakes')
else 
    fprintf('are mistakes')
end
