#include<iostream>
#include<cmath>
#include<vector>
#include<random>
#include"omp.h"

double function(double x, double mu, double sigma);
std::vector<double> metropolis(int N,double mu, double sigma);

int main(int argc, char **argv)
{
  int N = atoi(argv[1]);
  double mu = atof(argv[2]);
  double sigma = atof(argv[3]);
  std::cout.precision(16);
  std::cout.setf(std::ios::scientific);
  
  // std::vector<double> lista(N);
  
  double lista[N];
  // Inicializar
  for (int ii = 0; ii < N; ii++) {
    lista[ii]=0.0; 
  }
  // for (int ii = 0; ii < N; ii++) {
  //   std::cout<<lista[ii]<<std::endl; 
  //  }
  
  Met(lista, N, mu, sigma);
  
  // We print only the 80% of the list
  int length = N*0.8;
  std::vector<double> distribution(length);
  for(int ii= N-length ;ii<N; ii++)
    {
      distribution[ii-N+length] = lista[ii];
      std::cout<<distribution[ii-N+length]<<std::endl;
      // printf("%.16f \n",distribution[ii-N+length]);
    }
  
  // for (int ii = 0; ii < N; ii++) {
  //  std::cout<<lista[ii]<<std::endl; 
  // }
  
  // We print only the 80% of the list

  
  // lista = metropolis(N,mu,sigma);
  // double *ptr[N];
  // for (int ii = 0; ii < N; ii++) {
  //     ptr[ii] = &lista[ii]; // assign the address of integer.
  //  }
  // We print only the 80% of the list
  
  // int length = N*0.8;
  // std::vector<double> distribution(length);
  // for(int ii= N-length ;ii<N; ii++)
  //   {
  //     distribution[ii-N+length] = *ptr[ii];
  //     std::cout<<distribution[ii-N+length]<<std::endl;
  //     // printf("%.16f \n",distribution[ii-N+length]);
  //   }
  return 0;
}
double function(double x, double mu, double sigma)
{
  double gauss = (1.0/(sqrt(2.0*M_PI*sigma*sigma)))*exp(-(x-mu)*(x-mu)/(2.0*sigma*sigma));
  return gauss;
}
std::vector<double> metropolis(int N,double mu, double sigma)
{
  std::mt19937 generator(1);
  std::uniform_real_distribution<double> number(0.0,1.0);
  std::normal_distribution<double> Noise(0.0, 1.0);
  double propuesta = 0.0;
  std::vector<double> lista(N);
  lista[0] = number(generator);
  // Monte Carlo method
  for(int ii=1 ;ii<N; ii++)
    {
      propuesta = lista[ii-1] + Noise(generator);
      double ratio = function(propuesta,mu,sigma)/function(lista[ii-1],mu,sigma);
      double r = std::min(1.0,ratio);
      double alpha = number(generator);
      
      if(alpha < r)
	{
	  lista[ii] = propuesta;
	  continue;
	}
      
      else
	{
	  lista[ii] = lista[ii-1];
	  continue;
	}
    }
  return lista;
}

