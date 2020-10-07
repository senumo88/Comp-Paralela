#include <iostream>
#include <cstdlib>
#include <pthread.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <stdio.h>
#include <string.h>
#include <semaphore.h>


using namespace std;

#define thread_count 2
#define MAX 100

sem_t sems[thread_count];

char* strtok_r(char *str, const char *delim, char **nextp)
{
    char *ret;

    if (str == NULL)
    {
        str = *nextp;
    }

    str += strspn(str, delim);

    if (*str == '\0')
    {
        return NULL;
    }

    ret = str;

    str += strcspn(str, delim);

    if (*str)
    {
        *str++ = '\0';
    }

    *nextp = str;

    return ret;
}


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
        my_string = strtok_r(my_line, " \t\n", &saveptr);
        while ( my_string != NULL )
        {
            count++;
            printf("Thread %ld > string %d = %s\n", my_rank, count, my_string);
            my_string = strtok_r(NULL, " \t\n", &saveptr);
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
