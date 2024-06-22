#pragma once

#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <iostream>

// typedef float64 psi_dtype;
//typedef float psi_dtype;

#define CUDA_TRY(call)                                                          \
  do {                                                                          \
    cudaError_t const status = (call);                                          \
    if (cudaSuccess != status) {                                                \
      printf("%s %s %d\n", cudaGetErrorString(status), __FILE__, __LINE__);  \
    }                                                                           \
  } while (0)


template <size_t N>
struct KeyT {
    char data[N];

    __device__ __host__ constexpr size_t getKeyTSize() const {
        return N;
    }

    __device__ __host__ KeyT() {
        std::fill(data, data + N, 0);
    }

    __device__ __host__ KeyT(const char (&arr)[N]) {
        std::copy(arr, arr + N, data);
    }

    __device__ __host__ KeyT(uint64_t value) {
        uint64_t* ptr = static_cast<uint64_t *>((void*)data);
        for (int i = 0; i < N/8; i++)
            ptr[i] = value;
    }

    __device__ __host__ KeyT(const uint64_t *values) {
        uint64_t* ptr = static_cast<uint64_t *>((void*)data);
        for (int i = 0; i < N/8; i++)
            ptr[i] = values[i];
    }

    __device__ __host__ bool operator==(const KeyT<N>& other) const {
        const uint64_t *pOther = reinterpret_cast<const uint64_t*>(other.data);
        const uint64_t *pThis = reinterpret_cast<const uint64_t*>(data);
        for (size_t i = 0; i < N/8; ++i) {
            if (pThis[i] != pOther[i]) {
                return false;
            }
        }
        return true;
        //return std::memcmp(data, other.data, N) == 0;
    }

    __device__ __host__ bool operator<(const KeyT<N>& other) const {
        const uint64_t *pOther = reinterpret_cast<const uint64_t*>(other.data);
        const uint64_t *pThis = reinterpret_cast<const uint64_t*>(data);
        for (size_t i = 0; i < N/8; ++i) {
            if (pThis[i] < pOther[i]) {
                return true;
            } else if (pThis[i] > pOther[i]) {
                return false;
            }
        }
        return false;
        //return std::lexicographical_compare(data, data + N, other.data, other.data + N);
    }
};

struct ValueT{
    psi_dtype data[2];
};

template<size_t N>
__inline__ __device__ __host__ int myHashFunc(KeyT<N> value, int threshold) {
    //BKDR hash
    unsigned int seed = 31;
    char* values = static_cast<char*>(value.data);
    int len = value.getKeyTSize();
    unsigned int hash = 171;
    while(len--) {
        char v = (~values[len-1])*(len&1) + (values[len-1])*(~(len&1));
        hash = hash * seed + (v&0xF);
    }
    return (hash & 0x7FFFFFFF) % threshold;
}

template<int len, uint64_t seed=0xCBF29CE484222325>
__inline__ __device__ __host__ uint64_t murmurhash3_64(const char *key) {
    const uint64_t m = 0xC6A4A7935BD1E995;
    const int r = 47;

    uint64_t h = seed ^ (len * m);

    const uint64_t* data = reinterpret_cast<const uint64_t*>(key);
    const uint64_t* end = data + (len / 8);

    constexpr int nblocks = len / 8;
    const uint64_t* blocks = (const uint64_t*) key;
    #pragma unroll
    for (int i = 0; i < nblocks; i++) {
        uint64_t k = blocks[i];
        k *= m;
        k ^= k >> r;
        k *= m;

        h ^= k;
        h *= m;
    }

    const unsigned char* data2 = reinterpret_cast<const unsigned char*>(end);

    switch(len & 7) {
        case 7: h ^= uint64_t(data2[6]) << 48;
        case 6: h ^= uint64_t(data2[5]) << 40;
        case 5: h ^= uint64_t(data2[4]) << 32;
        case 4: h ^= uint64_t(data2[3]) << 24;
        case 3: h ^= uint64_t(data2[2]) << 16;
        case 2: h ^= uint64_t(data2[1]) << 8;
        case 1: h ^= uint64_t(data2[0]);
                h *= m;
    };

    h ^= h >> r;
    h *= m;
    h ^= h >> r;

    return h;
}


template<int LEN, uint32_t seed=7293>
__inline__ __device__ __host__ uint32_t murmurhash3_32(const char* key) {
    constexpr uint32_t c1 = 0xcc9e2d51;
    constexpr uint32_t c2 = 0x1b873593;
    constexpr uint32_t r1 = 15;
    constexpr uint32_t r2 = 13;
    constexpr uint32_t m = 5;
    constexpr uint32_t n = 0xe6546b64;
    uint32_t hash = seed;

    constexpr int nblocks = LEN / 4;
    const uint32_t* blocks = (const uint32_t*) key;
    int i;

    #pragma unroll
    for (i = 0; i < nblocks; i++) {
        uint32_t k = blocks[i];
        k *= c1;
        k = (k << r1) | (k >> (32 - r1));
        k *= c2;

        hash ^= k;
        hash = ((hash << r2) | (hash >> (32 - r2))) * m + n;
    }

    const uint8_t* tail = (const uint8_t*)(key + nblocks * 4);
    uint32_t k1 = 0;

    switch (LEN & 3) {
        case 3: k1 ^= tail[2] << 16;
        case 2: k1 ^= tail[1] << 8;
        case 1: k1 ^= tail[0];
                k1 *= c1; k1 = (k1 << r1) | (k1 >> (32 - r1)); k1 *= c2; hash ^= k1;
    };

    hash ^= LEN;
    hash ^= (hash >> 16);
    hash *= 0x85ebca6b;
    hash ^= (hash >> 13);
    hash *= 0xc2b2ae35;
    hash ^= (hash >> 16);

    return hash;
}

__inline__ __device__ __host__ uint64_t hash64(const char *str, const int len) {
    const uint64_t prime = 0x100000001B3;
    const uint64_t seed = 0xCBF29CE484222325;
    uint64_t hash = seed;

    for (int i = 0; i < len; i++) {
        hash = (hash ^ str[i]) * prime;
    }

    return hash;
}

__inline__ __device__ __host__ int hash64Func1(char *values, const int len) {
    int p = 16777619;
    int hash = (int)216161L;
    #pragma unroll
    for (int i = 0; i < len; i ++)
        hash = (hash ^ values[i]) * p;
    hash += hash << 13;
    hash ^= hash >> 7;
    hash += hash << 3;
    hash ^= hash >> 17;
    hash += hash << 5;
    return hash & 0x7FFFFFFF;
}

template<size_t N>
__inline__ __device__ __host__ int hash64Func(KeyT<N> value, int threshold) {    
    constexpr int len = value.getKeyTSize();
    char *values = static_cast<char*>(value.data);
    //return static_cast<int>(hash64(values, len) % threshold);
    //return int(hash64Func1(values, len) % threshold);
    //return murmurhash3_32(values, len) % threshold;
    //return murmurhash3_32<len>(values) % threshold;
    return static_cast<int>(murmurhash3_64<len>(values) % threshold);
    //return murmurhash3_32<len, 16777619>(values) % threshold;
    //return fasthash64<len>((void*)values) % threshold;
}

template<size_t N>
__inline__ __device__ __host__ int hashFunc1(KeyT<N> value, int threshold) {
    int p = 16777619;
    int hash = (int)216161L;
    int len = value.getKeyTSize();
    char *values = static_cast<char*>(value.data);
    #pragma unroll
    for (int i = 0; i < len; i ++)
        hash = (hash ^ values[i]) * p;
    hash += hash << 13;
    hash ^= hash >> 7;
    hash += hash << 3;
    hash ^= hash >> 17;
    hash += hash << 5;
    return (hash & 0x7FFFFFFF) % threshold;
}

template<size_t N>
__inline__ __device__ __host__ int hashFunc2(KeyT<N> value, int threshold) {
    /*int len = sizeof(KeyT);
    char *values = static_cast<char*>(value.data);
    int hash = 324223113;
    for (int i = 0; i < len; i ++) 
        hash = (hash<<4)^(hash>>28)^values[i];
    return (hash & 0x7FFFFFFF) % threshold;*/

    unsigned int seed = 12313;
    char* values = static_cast<char*>(value.data);
    int len = value.getKeyTSize();
    unsigned int hash = 711371;
    #pragma unroll
    for (int i = len; i > 0; i --) {
        char v = (~values[i-1])*(i&1) + (values[i-1])*(~(i&1));
        hash = hash * seed + (v&0xF);
    }
    return (hash & 0x7FFFFFFF) % threshold;
}

template<size_t N>
__inline__ __device__ __host__ int hashFunc3(KeyT<N> value, int threshold) {
    char *values = static_cast<char*>(value.data);
    int b = 378551;
    int a = 63689;
    int hash = 0;
    int len = value.getKeyTSize();

    #pragma unroll
    for(int i = 0; i < len; i++) {
        hash = hash * a + values[i];
        a = a * b;
    }
    return (hash & 0x7FFFFFFF)%threshold;    
}

#define __hashFunc hashFunc1
//#define __hashFunc hash64Func

#define BFT uint32_t
template<size_t N>
struct myHashTable {
    KeyT<N>* keys;
    ValueT* values;
    int* bCount;
    BFT* bf;
    int bNum;
    int bSize;

    __inline__ __device__ int64_t search_key(KeyT<N> key) {
        //int hashvalue = myHashFunc(key, bNum);
        //int hashvalue = hashFunc1(key, bNum);
        int hashvalue = __hashFunc(key, bNum);
        int my_bucket_size = bCount[hashvalue];
        KeyT<N>* list = keys + (int64_t)hashvalue*bSize;
        int thre = sizeof(BFT)*8;
        BFT my_bf = bf[hashvalue];
        //if (//!((my_bf>>hashFunc1(key, thre))&1)
        if (!((my_bf>>hashFunc1(key, thre))&1)
                || !((my_bf>>hashFunc2(key, thre))&1)
                || !((my_bf>>hashFunc3(key, thre))&1)) 
        {
            return -1;
        }

        for (int i = 0; i < my_bucket_size; i ++) {
            if (list[i] == key) {
                return hashvalue*bSize + i;
            }
        }
        return -1;
    }
};

template<size_t N>
__global__ void build_hashtable_kernel(myHashTable<N> ht, KeyT<N>* all_keys, ValueT* all_values, int ele_num, int* build_failure) {
    int bucket_num = ht.bNum;
    int bucket_size = ht.bSize;
    KeyT<N>* keys = ht.keys;
    ValueT* values = ht.values;
    int* bucket_count = ht.bCount;
    int thread_idx = blockIdx.x*blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    for (int i = thread_idx; i < ele_num; i = i+total_threads) {
        KeyT<N> my_key = all_keys[i];
        ValueT my_value = all_values[i];
        //int hashed_value = myHashFunc(my_key, bucket_num);
        //int hashed_value = hashFunc1(my_key, bucket_num);
        //int hashed_value = __hashFunc(my_key, bucket_num);
        int hashed_value = __hashFunc<N>(my_key, bucket_num);
        int write_off = atomicAdd(bucket_count + hashed_value, 1);
        if (write_off >= bucket_size) {
            build_failure[0] = 1;
            //printf("keyIdx is %d, hashed value is %d, now size is %d, error\n", i, hashed_value, write_off);
            break;
        }
        keys[hashed_value*bucket_size + write_off] = my_key;
        values[hashed_value*bucket_size + write_off] = my_value;
    }
    return ;
}

template<size_t N>
__global__ void build_hashtable_bf_kernel(myHashTable<N> ht) {
    int bucket_num = ht.bNum;
    int bucket_size = ht.bSize;
    KeyT<N>* keys = ht.keys;
    int* bucket_count = ht.bCount;
    int thread_idx = blockIdx.x*blockDim.x + threadIdx.x;
    for (int bid = thread_idx; bid < bucket_num; bid += gridDim.x * blockDim.x) {
        int my_bsize = bucket_count[bid];
        BFT my_bf = 0;
        for (int e = 0; e < my_bsize; e ++) {
            KeyT<N> my_value = keys[bid * bucket_size + e];
            int hv = hashFunc1(my_value, sizeof(BFT)*8);
            my_bf |= (1<<hv);
            hv = hashFunc2(my_value, sizeof(BFT)*8);
            my_bf |= (1<<hv);
            hv = hashFunc3(my_value, sizeof(BFT)*8);
            my_bf |= (1<<hv);
        }
        ht.bf[bid] = my_bf;
    }
    return ;
}

template<size_t N>
void freeHashTable(myHashTable<N> ht) {
    CUDA_TRY(cudaFree(ht.keys));
    CUDA_TRY(cudaFree(ht.values));
    CUDA_TRY(cudaFree(ht.bCount));
    CUDA_TRY(cudaFree(ht.bf));
}

template<size_t N>
bool buildHashTable(myHashTable<N> &ht, KeyT<N>* all_keys, ValueT* all_values, int bucket_num, int bucket_size, int ele_num) {
    ht.bNum = bucket_num;
    ht.bSize = bucket_size;

    //printf("bnum is %d, bsize is %d, ele num is %d\n", bucket_num, bucket_size, ele_num);

    int total_size = ht.bNum * ht.bSize;
    CUDA_TRY(cudaMalloc((void **)&ht.keys, sizeof(KeyT<N>)*total_size));
    CUDA_TRY(cudaMalloc((void **)&ht.values, sizeof(ValueT)*total_size));
    CUDA_TRY(cudaMalloc((void **)&ht.bCount, sizeof(int)*bucket_num));
    CUDA_TRY(cudaMalloc((void **)&ht.bf, sizeof(BFT)*bucket_num));
    CUDA_TRY(cudaMemset(ht.bCount, 0, sizeof(int)*bucket_num));
    CUDA_TRY(cudaMemset(ht.bf, 0, sizeof(BFT)*bucket_num));
    
    int* build_failure;
    CUDA_TRY(cudaMalloc((void **)&build_failure, sizeof(int)));
    CUDA_TRY(cudaMemset(build_failure, 0, sizeof(int)));

    #ifdef DEBUG_LOCAL_ENERGY
        double total_bytes = sizeof(KeyT<N>)*total_size + sizeof(ValueT)*total_size + sizeof(int)*bucket_num + 
                                sizeof(BFT)*bucket_num + sizeof(int)*bucket_num + sizeof(BFT)*bucket_num;
        printf("BloomHash table memory occupied: %.4f MB\n", total_bytes / 1024 / 1024);
    #endif

    //build hash table kernel
    //TODO: here we use atomic operations for building hash table for simplicity.
    //If we need better performance for this process, we can use multi-split.

    cudaEvent_t start, stop;
    float esp_time_gpu;
    CUDA_TRY(cudaEventCreate(&start));
    CUDA_TRY(cudaEventCreate(&stop));
    CUDA_TRY(cudaEventRecord(start, 0));

    int block_size = 256;
    int block_num = 2048;
    build_hashtable_kernel<<<block_num, block_size>>>(ht, all_keys, all_values, ele_num, build_failure);
    CUDA_TRY(cudaDeviceSynchronize());
    build_hashtable_bf_kernel<<<block_num, block_size>>>(ht);
    CUDA_TRY(cudaDeviceSynchronize());

    CUDA_TRY(cudaEventRecord(stop, 0));
    CUDA_TRY(cudaEventSynchronize(stop));
    CUDA_TRY(cudaEventElapsedTime(&esp_time_gpu, start, stop));
    //printf("Time for build_hashtable_kernel is: %f ms\n", esp_time_gpu);

    /*int* h_hash_count = new int[bucket_num];
    cudaMemcpy(h_hash_count, ht.bCount, sizeof(int)*bucket_num, cudaMemcpyDeviceToHost);
    for (int i = 0; i < bucket_num; i ++)
        printf("%d ", h_hash_count[i]);
    printf("\n");
    delete [] h_hash_count;*/

    /*KeyT *h_keys = new KeyT[bucket_num*bucket_size];
    cudaMemcpy(h_keys, ht.keys, sizeof(KeyT)*bucket_size*bucket_num, cudaMemcpyDeviceToHost);
    printf("here is the bucket:\n");
    for (int i = 0; i < bucket_num; i ++) {
        printf("bucket %d:\n", i);
        for (int j = 0; j < h_hash_count[i]; j ++) {
            h_keys[i*bucket_size + j].print(0);
        }
    }
    printf("\n");
    delete [] h_keys;*/

    //build success check
    int* build_flag = new int[1];
    CUDA_TRY(cudaMemcpy(build_flag, build_failure, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaDeviceSynchronize());
    bool return_state = build_flag[0] == 0 ? true : false;
    if (build_flag[0] == 1) {
        CUDA_TRY(cudaFree(ht.keys));
        CUDA_TRY(cudaFree(ht.values));
        CUDA_TRY(cudaFree(ht.bCount));
        CUDA_TRY(cudaFree(ht.bf));
    } else {
        //printf("build hash table success\n");
    }
    delete [] build_flag;
    CUDA_TRY(cudaFree(build_failure));
    return return_state;
}

/*
int main() {
    KeyT<16> key1(0x34567890ABCDEF);
    uint64_t arr[2] = {0x1234567890ABCDEF, 0xFEDCBA0987654321};
    KeyT<16> key2(arr);

    bool areEqual = (key1 == key2);
    std::cout << "Keys are " << (areEqual ? "equal" : "not equal") << std::endl;
    
    bool isFirstLess = (key1 < key2);
    std::cout << "Key1 is " << (isFirstLess ? "less than" : "not less than") << " Key2" << std::endl;

    return 0;
}
*/
