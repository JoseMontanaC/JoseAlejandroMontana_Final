#include<iostream>
#include<cmath>
#include<vector>
#include<random>
#include "omp.h"

double function(double x, double mu, double sigma);
void Met(double *lista, int N,double mu, double sigma,int thread,int seed);

int main(int argc, char **argv)
{
  int N = atoi(argv[1]);
  double mu = atof(argv[2]);
  double sigma = atof(argv[3]);
  std::cout.precision(16);
  std::cout.setf(std::ios::scientific);
  /* Archivo para escribir */
  FILE *out;
  
  double lista[N];
  // Inicializar
  for (int ii = 0; ii < N; ii++) {
    lista[ii]=0.0; 
  }
  
 #pragma omp parallel for
  for(int i=1;i<=8;i++)
    {
      // printf("voy en el %d \n",i);
      Met(lista, N, mu, sigma,i,i);
    }
 
  
 
  return 0;
}
double function(double x, double mu, double sigma)
{
  double gauss = (1.0/(sqrt(2.0*M_PI*sigma*sigma)))*exp(-(x-mu)*(x-mu)/(2.0*sigma*sigma));
  return gauss;
}


void Met(double *lista,int N,double mu, double sigma,int thread,int seed)
{
  FILE *out;
  char filename[128];
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> number(0.0,1.0);
  std::normal_distribution<double> Noise(0.0, 1.0);
  double propuesta = 0.0;
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
  sprintf(filename, "cadena_%d.txt",thread);
  if(!(out = fopen(filename, "w")))
    {
      fprintf(stderr, "Problema abriendo el archivo\n");
      exit(1);
    }
   for(int i=0;i<N;i++)
    {
      fprintf(out, "%.5f \n",lista[i] );
    }
  fclose(out);
}
