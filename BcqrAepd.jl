module compositeQR

export cqr

using RCall

function cqr(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, iter::Int, burn::Int, libLoc::String = "C:/Users/lukar818/Documents/R/win-library/4.0")
  rcopy(R"""

  y = $y
  X = $X
  n.sampler = $iter
  n.burn = $burn
  libLoc = $libLoc
  thin = 1

  library(SuppDists, lib.loc=libLoc)
  library(mvtnorm, lib.loc=libLoc)
  library(gtools, lib.loc=libLoc)

  cqr.lasso = function(x,y,K=9,n.sampler=13000,n.burn=3000,thin=20) {
                           #	x:		the design matrix
                           #	y:		the response
                           #	n.sample:	the length of the Markov chain
                           #	n.burn:	the length of burn-in

    theta = (1:K)/(K+1)
    n = dim(x)[1]
    p = dim(x)[2]
    xi1 = (1-2*theta)/(theta*(1-theta))
    xi2 = sqrt(2/(theta*(1-theta)))
    eps = y-x%*%solve(t(x)%*%x)%*%t(x)%*%y

                                          #---	Priors
    a = 1e-1
    b = 1e-1
    c = 1e-1
    d = 1e2

    ##The parameters with ".c" are the temporary ones that we use for updating.
    ##The parameters with ".p" are the recorded ones.
    ##Initialization

    alpha.c = rep(1,K)
    zi.c = apply(rmultinom(n,1,alpha.c),2,which.max)
    xi1.c = xi1[zi.c]
    xi2.c = xi2[zi.c]
    pi.c = rep(1/K,K)
    beta.c = rep(1,p)
    tz.c = rep(1,n)
    s.c = rep(1,p)
    tau.c = 1
    eta2.c = 1
    b.k = quantile(eps,prob=theta)
    b.c = b.k[zi.c]

                                          #---	Iteration
    zi.p = matrix(0,n.sampler,n)
    pi.p = matrix(1/K,n.sampler,K)
    beta.p = matrix(0, n.sampler, p)
    tz.p = matrix(0, n.sampler, n)
    b.p = matrix(0, n.sampler, K)
    tau.p = eta2.p = dic.p = rep(0, n.sampler)
    for(iter in 1:n.sampler){
      if(iter/1000 == as.integer(iter/1000)) {
        #print(paste("This is step ", iter, sep=""))
      }

                                          #---	The full conditional for tz
      temp.lambda = xi1.c^2*tau.c/(xi2.c^2)+2*tau.c
      temp.nu =  sqrt(temp.lambda*xi2.c^2/(tau.c*(y-b.c-x%*%beta.c)^2))
      index = 1:n
      flag = 1
      while(flag){
        temp.tz = rinvGauss(length(index),lambda=temp.lambda,nu=temp.nu[index])
        flag = any(temp.tz<=0)|any(is.na(temp.tz))
        tz.c[index[temp.tz>0]] = 1/temp.tz[temp.tz>0]
        index = setdiff(index,index[temp.tz>0])
      }

                                          #---	The full conditional for s
      temp.lambda = eta2.c
      temp.nu =  sqrt(temp.lambda/beta.c^2)
      index = 1:p
      flag = 1
      while(flag){
        temp.s = rinvGauss(length(index),lambda=temp.lambda,nu=temp.nu[index])
        flag = any(temp.s<=0)|any(is.na(temp.s))
        s.c[index[temp.s>0]] = 1/temp.s[temp.s>0]
        index = setdiff(index,index[temp.s>0])
      }

                                          #---	The full conditional for beta
      for(k in 1:p){
        temp.var = (sum(x[,k]^2*tau.c/(xi2.c^2*tz.c))+1/s.c[k])^(-1)
        temp.mean = sum(x[,k]*(y-b.c-xi1.c*tz.c-x[,-k]%*%beta.c[-k])*tau.c/(xi2.c^2*tz.c))*temp.var
        beta.c[k] = rnorm(1,mean=temp.mean,sd=sqrt(temp.var))
      }

                                          #---	The full conditional for tau
      temp.shape = a+3/2*n
      temp.rate = sum((y-b.c-x%*%beta.c-xi1.c*tz.c)^2/(2*xi2.c^2*tz.c)+tz.c)+b
      tau.c = rgamma(1,shape=temp.shape,rate=temp.rate)

                                          #---	The full conditional for eta2
      temp.shape = p+c
      temp.rate = sum(s.c)/2+d
      eta2.c = 1 #rgamma(1,shape=temp.shape,rate=temp.rate)

                                          #---	The full conditional for zi
      for(i in 1:n){
        temp.power = (y[i]-b.k-sum(x[i,]*beta.c)-xi1*tz.c[i])^2*tau.c/(xi2^2*tz.c[i])
        temp.alpha = pi.c*exp(-0.5*temp.power)/xi2
        zi.c[i] = which.max(rmultinom(1,1,temp.alpha/sum(temp.alpha)))
      }
      xi1.c = xi1[zi.c]
      xi2.c = xi2[zi.c]

                                          #---	The full conditional for pi
      n.c = rep(0,K)
      for(k in 1:K){
        if(!is.null(which(zi.c==k))){
          n.c[k] = length(which(zi.c==k))
        }
      }
      pi.c = rdirichlet(1,n.c+alpha.c)

                                          #---	The full conditional for b
      dic.c = 0
      for(k in 1:K){
        which.k = which(zi.c==k)
        if(length(which.k)>0){
          bc = y[which.k]-x[which.k,]%*%beta.c-xi1.c[which.k]*tz.c[which.k]
          sc = tau.c/(xi2.c[which.k]^2*tz.c[which.k])
          mean = sum(bc*sc)/sum(sc)
          sd = 1/sqrt(sum(sc))
          truncated <- 0
          if(truncated){
            if(k==1){
              b.k[k] = rtnorm(1,mean=mean,sd=sd,upper=b.k[k+1])
            }else if(k==K){
              b.k[k] = rtnorm(1,mean=mean,sd=sd,lower=b.k[k-1])
            }else{
              b.k[k] = rtnorm(1,mean=mean,sd=sd,lower=b.k[k-1],upper=b.k[k+1])
            }
          }else{
            b.k[k] = rnorm(1,mean=mean,sd=sd)
          }
          dic.temp = sum((y[which.k]-b.k[k]-x[which.k,]%*%beta.c-xi1.c[which.k]*tz.c[which.k])^2*tau.c/xi2.c[which.k]^2/tz.c[which.k])
          dic.temp = dic.temp + 2*sum(log(2*pi*xi2.c[which.k]^2*tz.c[which.k]/tau.c))
          uu = y[which.k]-b.k[k]-x[which.k,]%*%beta.c
          uu[which(uu>=0)] = theta[k]*uu[which(uu>=0)]
          uu[which(uu<0)] = (theta[k]-1)*uu[which(uu<0)]
          dic.temp = 2*sum(tau.c*uu-log(theta[k]*(1-theta[k])*tau.c))
          dic.c = dic.c + dic.temp
        }
      }
      b.c = b.k[zi.c]

      tz.p[iter,] = tz.c
      beta.p[iter,] = beta.c
      tau.p[iter] = tau.c
      eta2.p[iter] = eta2.c
      zi.p[iter,] = zi.c
      pi.p[iter,] = pi.c
      b.p[iter,] = b.k

      dic.p[iter] = dic.c/n

    }
    temp = seq(1,n.sampler-n.burn, by=thin)
    result = list(beta = beta.p[-(1:n.burn),][temp,],
      tau = tau.p[-(1:n.burn)][temp],
      eta2 = eta2.p[-(1:n.burn)][temp],
      tz = tz.p[-(1:n.burn),][temp,],
      pi = pi.p[-(1:n.burn),][temp,],
      zi = zi.p[-(1:n.burn),][temp,],
      b = b.p[-(1:n.burn),][temp,],
      dic = dic.p[-(1:n.burn)][temp])

    return(result)
  }

  cqr.lasso(X, y, 9, n.sampler, n.burn, thin)
  """)
end

end
