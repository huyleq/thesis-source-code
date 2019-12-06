function uv=myconv(u,v,zeroIndexV)
    n=size(u,1);
    m=size(v,1);
    nNegLag=zeroIndexV-1;
    nPosLag=m-nNegLag-1;
    A=zeros(n,m);
    for i=1:n
        for j=1:m
            k=i-j+nNegLag+1;
            if(k>0 && k<=n)
                A(i,j)=u(k);
            end
        end
    end
    uv=A*v;
end
