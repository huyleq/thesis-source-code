function [eps,del]=epsDelFromSmecFr(epsfile,delfile,smecfile,smecfrAtMaxAniso,epsAtMaxAniso,delAtMaxAniso)
    [nx,ox,dx]=get_par3(smecfile,'n1','o1','d1');
    [nz,oz,dz]=get_par3(smecfile,'n2','o2','d2');
    smecfr=sepread(smecfile,nx,nz);
    eps=zeros(nx,nz+1); 
    del=zeros(nx,nz+1); 
    for i=1:nx
        [~,j]=min(abs(smecfr(i,:)-smecfrAtMaxAniso));
        eps(i,2:j)=linspace(0,epsAtMaxAniso,j-1);
        eps(i,j+1:end)=epsAtMaxAniso;
        del(i,2:j)=linspace(0,delAtMaxAniso,j-1);
        del(i,j+1:end)=delAtMaxAniso;
    end
    sepwrite(epsfile,eps,[nx;nz+1],[ox;oz],[dx;dz]);
    sepwrite(delfile,del,[nx;nz+1],[ox;oz],[dx;dz]);
end
