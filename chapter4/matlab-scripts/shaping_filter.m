function filter=shaping_filter(original,desired,zeroIndex,nPosLag,nNegLag)
    filter_len=nPosLag+nNegLag+1;
    n=size(original,1);
    A=zeros(n,filter_len);

    for i=1:n
        for j=1:filter_len
            k=i-j+nNegLag+1;
            if(k>0 && k<=n)
                A(i,j)=original(k);
            end
        end
    end
    filter=linsolve(transpose(A)*A,transpose(A)*desired);
    filtered=myconv(original,filter,nNegLag+1);
    
    x=linspace(-zeroIndex+1,n-zeroIndex,n);
    figure
    hold on
    plot(x,desired,'Linewidth',2.5)
    plot(x,filtered,'Linewidth',2.5)
    hold off
    legend('desired','filter*original')
    fprintf('mean squared error %f\n',sum((desired-filtered).^2));

    figure
    plot(linspace(-nNegLag,nPosLag,filter_len),filter,'LineWidth',2.5)
end
