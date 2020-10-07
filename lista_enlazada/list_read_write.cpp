#include <iostream>
#include <cstdlib>
#include <pthread.h>
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace std;

#define thread_count 2
#define insert_per 80
#define delete_per 10
#define member_per 10
#define n 100000

struct list_node_s
{
    int data;
    struct list_node_s* next;
};

pthread_rwlock_t rwlock;

int Member(int value, struct list_node_s* head_p)
{
    struct list_node_s* curr_p = head_p;

    while (curr_p != NULL && curr_p->data < value)
    {
        curr_p = curr_p->next;
    }

    if (curr_p == NULL || curr_p->data > value)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}


int Insert(int value, struct list_node_s** head_p)
{
    struct list_node_s* curr_p = *head_p;
    struct list_node_s* pred_p = NULL;
    struct list_node_s* temp_p;

    while(curr_p != NULL && curr_p->data < value)
    {
        pred_p = curr_p;
        curr_p = curr_p->next;
    }

    if (curr_p == NULL || curr_p->data > value)
    {
        temp_p = (list_node_s*)malloc(sizeof(struct list_node_s));
        temp_p->data = value;
        temp_p->next = curr_p;
        if (pred_p == NULL)
        {
            *head_p = temp_p;
        }
        else
        {
            pred_p->next = temp_p;
        }
        return 1;
    }
    else
    {
        return 0;
    }
}

int Delete(int value, struct list_node_s** head_p)
{
    struct list_node_s* curr_p = *head_p;
    struct list_node_s* pred_p = NULL;

    while (curr_p != NULL && curr_p->data < value)
    {
        pred_p = curr_p;
        curr_p = curr_p->next;
    }

    if (curr_p != NULL && curr_p->data == value)
    {
        if (pred_p == NULL)
        {
            *head_p = curr_p->next;
            free(curr_p);
        }
        else
        {
            pred_p->next = curr_p->next;
            free(curr_p);
        }
        return 1;
    }
    else
    {
        return 0;
    }
}

///LISTA GLOBAL
list_node_s mylist;


//funcion de control
void* Thread_control(void* rank)
{
    long my_rank = (long) rank;
    double factor, my_sum = 0.0;
    long long i;
    long long my_n = n/thread_count;

    long long insert_count = my_n * insert_per / 100; //percentage of inserts
    long long delete_count = my_n * delete_per / 100; //percentage of delete
    long long member_count = my_n * delete_per / 100; //percentage of members

    //Inserts
    for (i = 0; i < insert_count; i++)
    {
        pthread_rwlock_rdlock(&rwlock);
        Insert(i*my_rank, &mylist.next);
        pthread_rwlock_unlock(&rwlock);
    }

    //Members
    for (i = 0; i < member_count; i++)
    {
        pthread_rwlock_rdlock(&rwlock);
        Member(i*my_rank, mylist.next);
        pthread_rwlock_unlock(&rwlock);
    }

    //Deletes
    for (i = 0; i < delete_count; i++)
    {
        pthread_rwlock_unlock(&rwlock);
        Delete(i*my_rank, &mylist.next);
        pthread_rwlock_unlock(&rwlock);
    }
    return NULL;
}

int main()
{
    pthread_rwlock_init(&rwlock, NULL);
    pthread_t threads[thread_count];

    long i;

    mylist.data = 0;

   //Tiempo de inicio
    auto start = std::chrono::high_resolution_clock::now();


    for(i= 0; i<thread_count; i++)
    {
        pthread_create(&threads[i], NULL, Thread_control, (void *)i);
    }

    for(i= 0; i<thread_count; i++)
    {
       pthread_join(threads[i], NULL);
    }

    free(threads);

    pthread_rwlock_destroy(&rwlock);

    //Tiempo de fin
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    cout << "Tiempo de ejecucion: " << elapsed.count() << " segundos" << endl << endl;

    return 0;
}
