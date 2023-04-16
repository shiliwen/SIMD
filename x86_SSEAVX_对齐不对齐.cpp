
#include<iostream>
#include <windows.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2
using namespace std;
const int N=1000;
const int p=10;
float m[N][N];
LARGE_INTEGER freq, t1, t2;

void m_reset()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < i; j++)
            m[i][j] = 0;
        m[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
            m[i][j] = rand();
    }
    for (int k = 0; k < N; k++)
        for (int i = k + 1; i < N; i++)
            for (int j = 0; j < N; j++)
                m[i][j] += m[k][j];
}
void normal()//普通
{
    double sumtime=0;
    QueryPerformanceFrequency(&freq);
    for(int x=0;x<p;x++)
    {
    m_reset();
    QueryPerformanceCounter(&t1);
    for (int k = 0; k < N; k++)
        {
                for (int j = k + 1; j < N; j++)
                {
                        m[k][j] = m[k][j] / m[k][k];

                }
                m[k][k] = 1.0;
                for (int i = k + 1; i < N; i++)
                {
                        for (int j = k + 1; j < N; j++)
                        {
                                m[i][j] -= m[i][k] * m[k][j];
                        }
                        m[i][k] = 0;
                }

        }
    QueryPerformanceCounter(&t2);
    sumtime += (t2.QuadPart - t1.QuadPart) * 1000.0 / freq.QuadPart;
     }
    cout << "normal_time: " << sumtime/p << "ms" << endl;
}
void unalign_sse()//不对齐SSE
{
    __m128 t5,t6,t3,t4;
    double sumtime=0;
    QueryPerformanceFrequency(&freq);
    for(int x=0;x<p;x++)
    {
        m_reset();
    QueryPerformanceCounter(&t1);
	for (int k = 0; k < N; k++) //除法优化
        {
        t5 = _mm_set1_ps(m[k][k]);
		int j = k + 1;
		for (; j + 4 < N; j += 4)
		{
			t6 = _mm_loadu_ps(&m[k][j]);
			t6 = _mm_div_ps(t6, t5);
			_mm_storeu_ps(&m[k][j], t6);
		}
		for (; j < N; j++)
			m[k][j] /= m[k][k];
		m[k][k] = 1.0;

		for (int i = k + 1; i < N; i++)
		{
			t5 = _mm_set1_ps(m[i][k]);
            int j = k + 1;
			for (; j + 4 < N; j += 4)
			{
				t6 = _mm_loadu_ps(&m[k][j]);
				t3 = _mm_loadu_ps(&m[i][j]);
				t4 = _mm_mul_ps(t5,t6);
				t3 = _mm_sub_ps(t3, t4);
				_mm_storeu_ps(&m[i][j], t3);
			}
			for (; j < N; j++)
				m[i][j] -= m[i][k] * m[k][j];
			m[i][k] = 0;
		}
	}
	QueryPerformanceCounter(&t2);
    sumtime += (t2.QuadPart - t1.QuadPart) * 1000.0 / freq.QuadPart;
    }
    cout << "unalign_sse_time: " << sumtime/p << "ms" << endl;
}


void align_sse()//对齐SSE
{

    float *A;
    A = (float*)_aligned_malloc(N * N * sizeof(float), 16);
	float(*a)[N] = (float(*)[N])A;
    QueryPerformanceFrequency(&freq);
    double sumtime=0;
    for(int x=0;x<p;x++)
    {
	for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < i; j++)
            a[i][j] = 0;
        a[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
            a[i][j] = rand();
    }
    for (int k = 0; k < N; k++)
        for (int i = k + 1; i < N; i++)
            for (int j = 0; j < N; j++)
                a[i][j] += a[k][j];

    __m128 t5,t6,t3,t4;
    QueryPerformanceCounter(&t1);
	for (int k = 0; k < N; k++) //除法优化
        {
        t5 = _mm_set1_ps(a[k][k]);
		int j = k + 1;
        for(int j = k + 1; j < k + 4 - k%4; j++) {
            a[k][j] = a[k][j]/a[k][k];
        }//串行计算至对齐
		for (; j + 4 <= N; j += 4)
		{
			t6 = _mm_load_ps(&a[k][j]);
			t6 = _mm_div_ps(t6, t5);
			_mm_storeu_ps(&a[k][j], t6);
		}
		for (; j < N; j++)
			a[k][j] /= a[k][k];
		m[k][k] = 1.0;

		for (int i = k + 1; i < N; i++)
		{
			t5 = _mm_set1_ps(a[i][k]);
            int j = k + 1;
            for(int j = k + 1; j < k + 4 - k%4; j++) {
                a[i][j] -=  a[i][k]*a[k][j];
            }
			for (; j + 4 <N; j += 4)
			{
				t6 = _mm_loadu_ps(&a[k][j]);
				t3 = _mm_loadu_ps(&a[i][j]);
				t4 = _mm_mul_ps(t5,t6);
				t3 = _mm_sub_ps(t3,t4);
				_mm_storeu_ps(&a[i][j], t3);
			}
			for (; j < N; j++)
				a[i][j] -= a[i][k] * a[k][j];
			a[i][k] = 0;
		}
	}
	QueryPerformanceCounter(&t2);
    sumtime += (t2.QuadPart - t1.QuadPart) * 1000.0 / freq.QuadPart;
    }
    cout << "align_sse_time: " << sumtime /p<< "ms" << endl;

}

void unalign_avx()//不对齐AVX
{
    __m256 temp1,temp2,temp3,temp4;
    double sumtime=0;
    QueryPerformanceFrequency(&freq);
    for(int x=0;x<p;x++)
    {
    m_reset();
    QueryPerformanceCounter(&t1);
	for (int k = 0; k < N; k++)
        {

		temp1 = _mm256_set1_ps(m[k][k]);
		int j = k + 1;
		for (; j + 8 < N; j += 8)
            {
			temp2 = _mm256_loadu_ps(&m[k][j]);
			temp2 = _mm256_div_ps(temp2, temp1);
			_mm256_storeu_ps(&m[k][j], temp2);
		}
		for (; j < N; j++)
			m[k][j] /= m[k][k];
		m[k][k] = 1.0;

		for (int i = k + 1; i < N; ++i)
		{
			int j = k + 1;
			 temp1 = _mm256_set1_ps(m[i][k]);
			for (; j + 8 < N; j += 8)
			{
                temp2 = _mm256_loadu_ps(&m[k][j]);
                temp3 = _mm256_loadu_ps(&m[i][j]);
				temp4 = _mm256_mul_ps(temp1, temp2);
				temp3 = _mm256_sub_ps(temp3, temp4);
				_mm256_storeu_ps(&m[i][j], temp3);
			}
			for (; j < N; ++j)
				m[i][j] -= m[i][k] * m[k][j];
			m[i][k] = 0;
		}

		}

	QueryPerformanceCounter(&t2);
    sumtime += (t2.QuadPart - t1.QuadPart) * 1000.0 / freq.QuadPart;
    }
    cout << "unalign_avx_time: " << sumtime /p<< "ms" << endl;
}

void align_avx()//对齐VX
{
    double sumtime=0;
    float *A;
    A = (float*)_aligned_malloc(N * N * sizeof(float), 32); // 分配32字节对齐的内存空间
	float(*a)[N] = (float(*)[N])A;
	__m256 temp1,temp2,temp3,temp4;
    QueryPerformanceFrequency(&freq);
    for(int x=0;x<p;x++)
    {
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < i; j++)
            a[i][j] = 0;
        a[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
            a[i][j] = rand();
    }
    for (int k = 0; k < N; k++)
        for (int i = k + 1; i < N; i++)
            for (int j = 0; j < N; j++)
                a[i][j] += a[k][j];

    QueryPerformanceCounter(&t1);
	for (int k = 0; k < N; k++)
        {
		temp1 = _mm256_set1_ps(a[k][k]);
		int j = k + 1;
		for(int j = k + 1; j < k + 8 - k%8; j++) {
            a[k][j] /= a[k][k];
        }//串行计算至对齐
		for (; j + 8 < N; j += 8) {
			temp2 = _mm256_loadu_ps(&a[k][j]);
			temp2 = _mm256_div_ps(temp2, temp1);
			_mm256_storeu_ps(&a[k][j], temp2);
		}
		for (; j < N; j++)
			a[k][j] /= a[k][k];
		a[k][k] = 1.0;
		for (int i = k + 1; i < N; ++i)
		{
			int j = k + 1;
			for(int j = k + 1; j < k + 8 - k%8; j++) {
                a[i][j] -=  a[i][k]*a[k][j];
            }
			 temp1 = _mm256_set1_ps(a[i][k]);
			for (; j + 8 < N; j += 8)
			{
                temp2 = _mm256_loadu_ps(&a[k][j]);
                temp3 = _mm256_loadu_ps(&a[i][j]);
				temp4 = _mm256_mul_ps(temp1, temp2);
				temp3 = _mm256_sub_ps(temp3, temp4);
				_mm256_storeu_ps(&a[i][j], temp3);
			}
			for (; j < N; ++j)
				a[i][j] -= a[i][k] * a[k][j];
			a[i][k] = 0;
		}
	}

	QueryPerformanceCounter(&t2);
sumtime += (t2.QuadPart - t1.QuadPart) * 1000.0 / freq.QuadPart;
}


    cout << "align_avx_time: " << sumtime /p<< "ms" << endl;

}

void print()
{
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
            cout<<m[i][j]<<" ";
        cout<<endl;
    }
}

int main()
{

normal();

align_sse();

unalign_sse();

unalign_avx();

align_avx();


return 0;
}
