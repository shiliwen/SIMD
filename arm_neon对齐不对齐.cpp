#include<iostream>
#include<arm_neon.h>
#include<sys/time.h>
#include<unistd.h>
#include<stdlib.h>


using namespace std;
const int N = 1024;
const int p=10;
float m[N][N];
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
void normal()
{
    double sumtime=0;
    struct timeval t1,t2,tresult;

    for(int x=0;x<p;x++)
    {
    m_reset();
	for (int k = 1; k < N; k++)
    {
		for (int j = k + 1; j < N; j++)
		     m[k][j] /= m[k][k];
		m[k][k] = 1.0;
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
				m[i][j] -=m[i][k] * m[k][j];
			m[i][k] = 0;
		}
	}
	gettimeofday(&t2, NULL);
	timersub(&t2, &t1, &tresult);

    sumtime += tresult.tv_sec*1000 + (1.0 * tresult.tv_usec)/1000;
    }
    cout << "normal_time: " << sumtime /p<< "ms" << endl;
}
void unalign_nemo()
{
    float32x4_t temp1,temp2,temp3,temp4;
    double sumtime=0;
    struct timeval t1,t2,tresult;

    for(int x=0;x<p;x++)
    {
    m_reset();
    gettimeofday(&t1, NULL);
	for (int k = 0; k < N; k++)
        {
		temp1 = vmovq_n_f32(m[k][k]);
		int j = k + 1;
		for (; j + 8 < N; j += 8) {
			temp2 = vld1q_f32(&m[k][j]);
			temp2 = vdivq_f32(temp2, temp1);
			vst1q_f32(&m[k][j], temp2);
		}
		for (; j < N; j++)
			m[k][j] /= m[k][k];
		m[k][k] = 1.0;
		for (int i = k + 1; i < N; ++i)
		{
			int j = k + 1;
			 temp1 = vmovq_n_f32(m[i][k]);
			for (; j + 8 < N; j += 8)
			{
                temp2 = vld1q_f32(&m[k][j]);
                temp3 = vld1q_f32(&m[i][j]);
				temp4 = vmulq_f32(temp1, temp2);
				temp3 = vsubq_f32(temp3, temp4);
				vst1q_f32(&m[i][j], temp3);
			}
			for (; j < N; ++j)
				m[i][j] -= m[i][k] * m[k][j];
			m[i][k] = 0;
		}
	}
	gettimeofday(&t2, NULL);
	timersub(&t2, &t1, &tresult);

    sumtime += tresult.tv_sec*1000 + (1.0 * tresult.tv_usec)/1000;
    }
    cout << "unalign_neon_time: " << sumtime /p<< "ms" << endl;
}

void align_neon()
{
    double sumtime=0;
    float *A;
    A = (float*)memalign(16,N * N * sizeof(float));
	float(*a)[N] = (float(*)[N])A;
	float32x4_t temp1,temp2,temp3,temp4;
    struct timeval t1,t2,tresult;
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
        for(int j = k + 1; j < k + 4 - k%4; j++)
            {
            a[k][j] /= a[k][k];
            }//串行计算至对齐
    gettimeofday(&t1, NULL);
	for (int k = 0; k < N; k++)
        {
		temp1 = vmovq_n_f32(a[k][k]);
		int j = k + 1;

		for (; j + 4 < N; j += 4) {
			temp2 = vld1q_f32(&a[k][j]);
			temp2 = vdivq_f32(temp2, temp1);
			vst1q_f32(&a[k][j], temp2);
		}
		for (; j < N; j++)
			a[k][j] /= a[k][k];
		a[k][k] = 1.0;
		for (int i = k + 1; i < N; ++i)
		{
			int j = k + 1;
			for(int j = k + 1; j < k + 4 - k%4; j++)
                {
                a[i][j] -= a[i][k]*a[k][j];
                }

			 temp1 = vmovq_n_f32(a[i][k]);
			for (; j + 4 < N; j += 4)
			{
                temp2 = vld1q_f32(&a[k][j]);
                temp3 = vld1q_f32(&a[i][j]);
				temp4 = vmulq_f32(temp1, temp2);
				temp3 = vsubq_f32(temp3, temp4);
				vst1q_f32(&a[i][j], temp3);
			}
			for (; j < N; ++j)
				a[i][j] -= a[i][k] * a[k][j];
			a[i][k] = 0;
		}
	}

	gettimeofday(&t2, NULL);
	timersub(&t2, &t1, &tresult);

    sumtime += tresult.tv_sec*1000 + (1.0 * tresult.tv_usec)/1000;
}
cout << "align_neon_time: " << sumtime /p<< "ms" << endl;

}

int main()
{
    normal();
    unalign_neon();
    align_neon();

}
