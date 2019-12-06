function [eta,del]=etaDelFromSmecFr(etafile,delfile,smecfile,smecfrAtMaxAniso,etaAtMaxAniso,delAtMaxAniso)
    [nx,ox,dx]=get_par3(smecfile,'n1','o1','d1');
    [nz,oz,dz]=get_par3(smecfile,'n2','o2','d2');
    smecfr=sepread(smecfile,nx,nz);
    eta=zeros(nx,nz+1); 
    del=zeros(nx,nz+1); 
    for i=1:nx
        [~,j]=min(abs(smecfr(i,:)-smecfrAtMaxAniso));
        eta(i,2:j)=linspace(0,etaAtMaxAniso,j-1);
        eta(i,j+1:end)=etaAtMaxAniso;
        del(i,2:j)=linspace(0,delAtMaxAniso,j-1);
        del(i,j+1:end)=delAtMaxAniso;
    end
    sepwrite(etafile,eta,[nx;nz+1],[ox;oz],[dx;dz]);
    sepwrite(delfile,del,[nx;nz+1],[ox;oz],[dx;dz]);
end
