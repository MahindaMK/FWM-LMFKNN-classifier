function out=intersec(a,b)

[n,m]=size(a);
for i=1:n
    out(i)=a(i)*b(i); %Using product as intersection
end
