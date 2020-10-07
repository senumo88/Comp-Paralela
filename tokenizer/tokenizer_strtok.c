#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <semaphore.h>

#define thread_count 2
#define MAX 100

sem_t sems[thread_count];

void* Tokenize(void* rank)
{
    long my_rank = (long) rank;
    int count;
    int next = (my_rank + 1) % thread_count;
    char *fg_rv;
    char my_line[MAX];
    char *my_string;
    char *saveptr;

    sem_wait(&sems[my_rank]);
    fg_rv = fgets(my_line, MAX, stdin);
    sem_post(&sems[next]);
    while (fg_rv != NULL)
    {
        printf("Thread %ld > my line = %s", my_rank, my_line);

        count = 0;
        my_string = strtok(my_line, " \t\n");
        while ( my_string != NULL )
        {
            count++;
            printf("Thread %ld > string %d = %s\n", my_rank, count, my_string);
            my_string = strtok(NULL, " \t\n");
        }

        sem_wait(&sems[my_rank]);
        fg_rv = fgets(my_line, MAX, stdin);
        sem_post(&sems[next]);
    }
    return NULL;
}

int main()
{
    pthread_t threads[thread_count];

    long i;


    for(i= 0; i<thread_count; i++)
    {
        pthread_create(&threads[i], NULL, Tokenize, (void *)i);
    }

    for(i= 0; i<thread_count; i++)
    {
       pthread_join(threads[i], NULL);
    }

    free(threads);

    return 0;
}
