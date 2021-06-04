#ifndef DEBUG_H_
#define DEBUG_H_

#include <iostream>

using std::cout;
using std::endl;

template <typename T>
void debug_array(const T *array, const int n)
{
    for (int i = 0; i < n; i++)
    {
        cout << array[i] << " ";
    }
    cout << endl;
}

template <typename T>
void debug_matrix(const T *mat, const int n, const int m)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            cout << mat[i * m + j] << " ";
        }
        cout << endl;
    }
}

#endif