#include <iostream>
#include <cstdlib>
#include <pthread.h>
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace std;

#define thread_count 4
#define m 8 //matrix m X n
#define n 8000000

int A[m][n]; //matrix
int x[n]; //vector
int y[m]; //Ax


void *Pth_mat_vect(void* rank)
{
    long my_rank = (long) rank;
    int i, j;
    int local_m = m/thread_count;
    int my_first_row = my_rank * local_m;
    int my_last_row = (my_rank+1) * local_m - 1;

    for (i = my_first_row; i <= my_last_row; i++)
    {
        y[i] = 0.0;
        for (j = 0; j < n; j++)
        {
            y[i] += A[i][j]*x[j];
        }
    }
    return NULL;
}



int main()
{
   pthread_t threads[thread_count];
   long i, j;

    //Iniciando A y x
    for(i=0; i<m; i++)
    {
        for(j=0; j<n; j++)
        {
            A[i][j] = i + j;
            x[j] = i - j;
        }
    }

   //Tiempo de inicio
    auto start = std::chrono::high_resolution_clock::now();

   for(i = 0; i < thread_count; i++ )
   {
      pthread_create(&threads[i], NULL, Pth_mat_vect, (void *)i);
   }

   for(i= 0; i<thread_count; i++)
   {
       pthread_join(threads[i], NULL);
   }

   free(threads);

   //Tiempo de fin
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    cout << "Tiempo de ejecucion: " << elapsed.count() << " segundos" << endl << endl;

    return 0;
}
