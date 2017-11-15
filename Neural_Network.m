%--------------------train data asssemble start-----------------%

M=dlmread('F:\4-2\Pattern Recognition Sessional\neural\dataFilesForNeuralNetwork\dataFilesForNeuralNetwork\trainNN.txt');
instances=M(:,1:size(M,2)-1);
classes=M(:,size(M,2):size(M,2));
newclasses=classes;
uniquestring=unique(classes);
uniquestring=transpose(uniquestring);
%----------------------train data assembling complete--------------------------%

%-----instance matrix normalization-----%
max_instance=max(instances,[],1);
for col=1:size(instances,2)
    for row=1:size(instances,1)
        instances(row,col)=instances(row,col)/max_instance(1,col);
    end
end


%-----instances should be normalized now----%

%-----instance matrix normalization-----%

%------------relevant arrays: instances, newclasses, uniquestrings -------------%

sigmoidA=0.1;
learning_rate=0.1;

%---------proti layer e koyta neuron ase sei info store kortesi---------%
layer_array=dlmread('F:\4-2\Pattern Recognition Sessional\neural\layer_config.txt');
%disp(layer_array);
%disp(size(layer_array));
new_layer_array=[size(instances,2),layer_array,size(uniquestring,2)];
disp('new layer array');
disp(new_layer_array);
%------------layer_array te ase, 1st element feature num, last elem
%possible class number------------%
%relevant array-------new_layer_array--------------%

%------network banano shuru-------%
%-------cell array lagbe mone hoy, parlam na use korte, too hard----%
network=zeros(max(new_layer_array),max(new_layer_array),size(new_layer_array,2)-1);
network_w_bias=zeros(size(new_layer_array,2)-1,max(new_layer_array));
max_layer_size=max(new_layer_array);
for i=2:size(new_layer_array,2)
    neuronmat=[];
    for j=1:max_layer_size;
        weightvector=[];
        for k=1:max_layer_size
            x=rand;
            weightvector=[weightvector,x];
        end
        w_bias=rand;
        network_w_bias((i-1),j)=w_bias;
        neuronmat=[neuronmat;weightvector];
    end
    network(:,:,(i-1))=neuronmat;
end
%-------network banano sesh-------%
%-------network e onek redundant jinis ase----access er somoy valo vabe
%handle korte hobe bepargula--------------------%
layer_size=size(network,3);
neuron_sigmoid_input=zeros(size(network,3),max(new_layer_array),size(instances,1));
neuron_sigmoid_output=zeros(size(network,3),max(new_layer_array),size(instances,1)); 
loss_value=zeros(size(network,3),max(new_layer_array),size(instances,1));
delta=zeros(size(network,3),max(new_layer_array),size(instances,1));
gradient_w=zeros(size(network,3),max(new_layer_array),size(instances,1));
%ei array teo kichu extra value thakbe
iter=0;
while 1
    %----------forward propagation korbo---%
    iter=iter+1;
    if iter>3000
        break;
    end
    Jerror=0.0;
    for ins=1:size(instances,1)
        curins=instances(ins,:);
        %disp(size(curins));
        featuresize=new_layer_array(1,1);
        for neur=1:featuresize
            neuron_sigmoid_output(1,neur,ins)=curins(1,neur);
            %disp(neuron_sigmoid_output(1,neur,ins));
        end
        for lyr=1:layer_size
            cur_layer=network(:,:,lyr);
            neuron_here=new_layer_array(1,lyr+1);
            prev_output_vector=neuron_sigmoid_output(lyr,:,ins);
            ss=0.0;
            for j=1:neuron_here
                weightvector=cur_layer(j,:);
                usedweight=new_layer_array(1,lyr);
                true_wvec=weightvector(1:usedweight);
                true_prev_output=prev_output_vector(1:usedweight);
                %disp('hello');
                %disp(true_wvec);
                %disp(true_prev_output);
                val=dot(true_wvec,true_prev_output);
                neuron_sigmoid_input(lyr+1,j,ins)=val+network_w_bias(lyr,j);
                neuron_sigmoid_output(lyr+1,j,ins)=1.0/(1.0+exp(-val*sigmoidA)); %---logistic function ta ber korte hobe
                %ss=ss+neuron_sigmoid_output(lyr+1,j,ins);
                %disp(val);
                %disp('hello2');
            end
            %for j=1:neuron_here
             %   neuron_sigmoid_output(lyr+1,j,ins)=neuron_sigmoid_output(lyr+1,j,ins)/ss;
            %end
            %---normalizing outputs---%
        end
        actual_class=newclasses(ins,1);
        outer_neuron=new_layer_array(1,size(new_layer_array,2));
        Eins=0.0;
        for out_neur=1:outer_neuron
            fvm=neuron_sigmoid_output(layer_size+1,out_neur,ins);
            ym=0.2;
            if(out_neur==actual_class)
                ym=0.8;
            end
            diff=fvm-ym;
            loss_value(layer_size+1,out_neur,ins)=diff;
            Eins=Eins+(diff*diff);
            derivative_fvm=sigmoidA*fvm*(1.0-fvm);
            delta(layer_size+1,out_neur,ins)=derivative_fvm*diff;
        end
        Eins=Eins/2.0;
        Jerror=Jerror+Eins;
    end
%-----------forward propagat hoise---khali 104 number line e change ante
%hobe-----%
%---------forward propagation sesh------%
    disp(Jerror);
    if Jerror<5.0
        break;
    end
    %------backward propagation korbo--------%
    for ins=1:size(instances,1)
        for lyr=layer_size:-1:2
            ei_layer_neuron=new_layer_array(1,lyr);
            porer_layer_neuron=new_layer_array(1,lyr+1);
            for ei_neuron=1:ei_layer_neuron
                ejrminus1=0.0;
                for porer_neuron=1:porer_layer_neuron
                    del=delta(lyr+1,porer_neuron,ins);
                    wkj=network(porer_neuron,ei_neuron,lyr);
                    ejrminus1=ejrminus1+(del*wkj);
                end
                loss_value(lyr,ei_neuron,ins)=ejrminus1;
                fvm=neuron_sigmoid_output(lyr,ei_neuron,ins);
                fderiv=sigmoidA*fvm*(1.0-fvm);
                delta(lyr,ei_neuron,ins)=fderiv*ejrminus1;
            end
        end
    end
    %------backward propagation sesh---------%
    
    %-----eibar weight update korte hobe------%
    for lyr=2:(layer_size+1)
        ei_layer_neuron=new_layer_array(1,lyr);
        ager_layer_neuron=new_layer_array(1,lyr-1);
        for ei_neuron=1:ei_layer_neuron
            %disp(new_layer_array(1,lyr-1));
            gradientwrj=zeros(1,new_layer_array(1,lyr-1));
            biasval=0.0;
            for ins=1:size(instances,1)
                deltaval=delta(lyr,ei_neuron,ins);
                biasval=biasval+deltaval;
                yvector=neuron_sigmoid_output(lyr-1,:,ins);
                myvector=yvector(:,1:size(gradientwrj,2));
                myvector=deltaval*myvector;
                gradientwrj=gradientwrj+myvector;
            end
            network_w_bias(lyr-1,ei_neuron)=network_w_bias(lyr-1,ei_neuron)-learning_rate*biasval;
            old_wvector=network(ei_neuron,:,lyr-1);
            gradientwrj(numel(old_wvector))=0;
            %disp(gradientwrj);
            new_wvector=old_wvector-learning_rate*gradientwrj;
            network(ei_neuron,:,lyr-1)=new_wvector;
        end
    end
    %-----eibar weight update korte hobe------%
end

%---------testing area---------------%

M_test=importdata('F:\4-2\Pattern Recognition Sessional\neural\dataFilesForNeuralNetwork\dataFilesForNeuralNetwork\testNN.txt');
instances_test=M(:,1:size(M,2)-1);
classes_test=M(:,size(M,2):size(M,2));
newclasses_test=classes_test;
uniquestring_test=unique(classes_test);

max_instance_test=max(instances_test,[],1);
for col=1:size(instances_test,2)
    for row=1:size(instances_test,1)
        instances_test(row,col)=instances_test(row,col)/max_instance_test(1,col);
    end
end

%------file er jinispati matrix e-----%

%uniquestring
disp(size(uniquestring,2));

confusion_matrix=zeros(size(uniquestring,2),size(uniquestring,2));

for ins=1:size(instances_test,1)
    curins=instances_test(ins,:);
    %disp(size(curins));
    featuresize=new_layer_array(1,1);
    for neur=1:featuresize
        neuron_sigmoid_output(1,neur,ins)=curins(1,neur);
        %disp(neuron_sigmoid_output(1,neur,ins));
    end
    for lyr=1:layer_size
        cur_layer=network(:,:,lyr);
        neuron_here=new_layer_array(1,lyr+1);
        prev_output_vector=neuron_sigmoid_output(lyr,:,ins);
        ss=0.0;
        for j=1:neuron_here
            weightvector=cur_layer(j,:);
            usedweight=new_layer_array(1,lyr);
            true_wvec=weightvector(1:usedweight);
            true_prev_output=prev_output_vector(1:usedweight);
            val=dot(true_wvec,true_prev_output);
            neuron_sigmoid_input(lyr+1,j,ins)=val+network_w_bias(lyr,j);
            neuron_sigmoid_output(lyr+1,j,ins)=1.0/(1.0+exp(-val*sigmoidA)); %---logistic function ta ber korte hobe
           % ss=ss+neuron_sigmoid_output(lyr+1,j,ins);
        end
        %for j=1:neuron_here
         %   neuron_sigmoid_output(lyr+1,j,ins)=neuron_sigmoid_output(lyr+1,j,ins)/ss; %---logistic function ta ber korte hobe
       % end
    end
    actual_class=newclasses_test(ins,1);
    outer_neuron=new_layer_array(1,size(new_layer_array,2));
    predict_class_idx=-1;
    neur_max_value=-100000000.00;
    for out_neur=1:outer_neuron
        fvm=neuron_sigmoid_output(layer_size+1,out_neur,ins);
        if fvm>neur_max_value
            predict_class_idx=out_neur;
            neur_max_value=fvm;
        end
    end
    disp(predict_class_idx);
    confusion_matrix(predict_class_idx,actual_class)=confusion_matrix(predict_class_idx,actual_class)+1;
end

%----confusion matrix creation complete----%

%-----code for testing accuracy, precision, recall----%

precision=0.0;
recall=0.0;

correct=0.0;

for classval=1:size(uniquestring,2)
    tp=confusion_matrix(classval,classval);
    fp=sum(confusion_matrix(:,classval))-confusion_matrix(classval,classval);
    fn=sum(confusion_matrix(classval,:),2)-confusion_matrix(classval,classval);
    pr=tp/(tp+fp);
    rc=tp/(tp+fn);
    precision=precision+pr;
    recall=recall+rc;
    correct=correct+tp;
end

precision=precision/(size(uniquestring,2));
recall=recall/(size(uniquestring,2));
total=sum(sum(confusion_matrix));
disp(size(total));
accuracy=correct/total;

accuracy=accuracy*100.0
precision=precision*100.0
recall=recall*100.0

%---------testing area----------------%