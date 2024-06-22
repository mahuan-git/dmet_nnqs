#pragma once

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <typeinfo>
#include <cstring>
#include <cassert>
#include <iomanip>      // std::setprecision


namespace npy_cpp {

typedef int8_t int8;
typedef int32_t int32;
typedef uint32_t uint32;
typedef int64_t int64;
typedef uint64_t uint64;
typedef float float32;
typedef double float64;

// Supported data types of save/load
enum DataType {
    CHAR, INT8, INT32, INT64, UINT64, FLOAT32, FLOAT64
};
const size_t DataTypeSize[] = {sizeof(char), sizeof(int8), sizeof(int32), sizeof(int64), sizeof(uint64), sizeof(float32), sizeof(float64)};
const char* DataTypeName[] = {"CHAR", "INT8", "INT32", "INT64", "UINT64", "FLOAT32", "FLOAT64"};

template <typename T>
DataType getDataType();

template <>
DataType getDataType<char>() {
    return DataType::CHAR;
}

template <>
DataType getDataType<int8>() {
    return DataType::INT8;
}

template <>
DataType getDataType<int32>() {
    return DataType::INT32;
}

template <>
DataType getDataType<int64>() {
    return DataType::INT64;
}

template <>
DataType getDataType<uint64>() {
    return DataType::UINT64;
}

template <>
DataType getDataType<float32>() {
    return DataType::FLOAT32;
}

template <>
DataType getDataType<float64>() {
    return DataType::FLOAT64;
}

// Basic structure of save/load
class ArrayInfo {
private:
    char* data;
    size_t length;
    DataType type;

public:
    ArrayInfo() : data(nullptr), length(0), type(DataType::INT32) {
        // std::cout << "construct empty" << "len:" << length << std::endl;
    }

    ArrayInfo(char *_data, size_t _length, DataType _type) : data(_data), length(_length), type(_type) {
        // std::cout << "construct munual" << std::endl;
    }

    template<typename T>
    ArrayInfo(T* array, size_t len) {
        length = len * sizeof(T);
        data = new char[length];
        std::memcpy(data, array, length);
        type = getDataType<T>();
    }

    ArrayInfo(const ArrayInfo& other)
        : length(other.length), type(other.type) {
        data = new char[length];
        std::memcpy(data, other.data, length);
        // std::cout << "copy construct" << std::endl;
    }

    ArrayInfo(ArrayInfo&& other) noexcept
        : data(other.data), length(other.length), type(other.type) {
        other.data = nullptr;
        other.length = 0;
        // std::cout << "move construct" << std::endl;
    }

    ArrayInfo& operator=(const ArrayInfo& other) {
        if (&other == this) return *this;
        delete[] data;

        length = other.length;
        type = other.type;
        data = new char[length];
        std::memcpy(data, other.data, length);
        // std::cout << "copy = construct" << std::endl;
        return *this;
    }

    ArrayInfo& operator=(ArrayInfo&& other) noexcept {
        if (&other == this) return *this;

        delete[] data;

        data = other.data;
        length = other.length;
        type = other.type;

        other.data = nullptr;
        other.length = 0;
        // std::cout << "move = construct" << std::endl;
        return *this;
    }

    template<typename T>
    T* getDataPtr() const {
        return reinterpret_cast<T*>(data);
    }

    size_t getArrayNum() const {
        return length / DataTypeSize[type];
    }

    char *getData() const {return data;}
    size_t getLength() const {return length;}
    DataType getType() const {return type;}
};

// Main manager class and interface
class NumpyCpp {
private:
    std::unordered_map<std::string, ArrayInfo> data;

public:
    ~NumpyCpp() {
        for (auto& pair : data) {
            delete[] pair.second.getData();
        }
    }

    void savez(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);

        for (const auto& pair : data) {
            size_t keySize = pair.first.size();
            file.write(reinterpret_cast<char*>(&keySize), sizeof(keySize));
            file.write(pair.first.c_str(), keySize);

            DataType type = pair.second.getType();
            file.write(reinterpret_cast<char*>(&type), sizeof(DataType));
            size_t length = pair.second.getLength();
            file.write(reinterpret_cast<char*>(&length), sizeof(length));
            file.write(pair.second.getData(), length);
            printf("[Save] key: %s len: %ld dtype: %s\n", pair.first.c_str(), length, DataTypeName[type]);
        }
        printf("Save data into file: %s\n", filename.c_str());
        file.close();
    }

    void loadz(const std::string& filename) {
        std::cout << "finame:" << filename << std::endl;
        std::ifstream file(filename, std::ios::binary);
        while (file.peek() != EOF) {
            size_t keySize;
            file.read(reinterpret_cast<char*>(&keySize), sizeof(keySize));

            char key[keySize + 1];
            file.read(key, keySize);
            key[keySize] = '\0';

            DataType type;
            file.read(reinterpret_cast<char*>(&type), sizeof(DataType));

            size_t length;
            file.read(reinterpret_cast<char*>(&length), sizeof(length));

            printf("[Load] key: %s len: %ld dtype: %s\n", key, length, DataTypeName[type]);

            char* dataArray = new char[length];
            file.read(dataArray, length);

            data[key] = ArrayInfo{dataArray, length, type};
        }
        file.close();
    }

    ArrayInfo& operator[](const std::string& key) {
        return data[key];
    }

    bool isExist(const std::string &key) {
        return data.find(key) == data.end() ? false : true;
    }

    template<typename T>
    void insert(const std::string& key, const T* array, size_t length) {
        T *noconst_array = const_cast<T *>(array);
        insert(key, noconst_array, length);
    }

    template<typename T>
    void insert(const std::string& key, T* array, size_t length) {
        char* dataArray = new char[length];
        std::memcpy(dataArray, array, length);
        DataType type = getDataType<T>();
        data[key] = ArrayInfo{dataArray, length, type};
    }
};

template<typename T>
void printArray(ArrayInfo &arrayInfo, std::string mess="") {
    std::cout << "Print array " << mess << ": ";
    for (size_t i = 0; i < arrayInfo.getArrayNum(); ++i) {
        if constexpr (std::is_same<T, int8_t>::value) {
            std::cout << static_cast<int>(arrayInfo.getDataPtr<T>()[i]) << " ";
        } else {
            std::cout << arrayInfo.getDataPtr<T>()[i] << " ";
        }
    }
    std::cout << "\n";
}

template<typename T>
void sumArray(ArrayInfo &arrayInfo, std::string mess="") {
    T sum = T(0), avg = T(0);
    for (size_t i = 0; i < arrayInfo.getArrayNum(); ++i) {
        sum += arrayInfo.getDataPtr<T>()[i];
    }
    avg = sum / arrayInfo.getArrayNum();
    std::cout << "Sum array " << mess << ": ";
    std::cout << "sum: " <<  std::setprecision(15) << sum << " avg: " <<  std::setprecision(15) << avg << " num: " << arrayInfo.getArrayNum() << "\n";
}

template<typename T>
void check(ArrayInfo &arrayInfo, T *arrDiff, size_t lenByte, std::string mess="", float64 eps=1e-12) {
//void check(ArrayInfo &arrayInfo, T *arrDiff, size_t lenByte, std::string mess="", float64 eps=1e-4) {
    assert(lenByte / sizeof(T) == arrayInfo.getArrayNum());
    for (size_t i = 0; i < arrayInfo.getArrayNum(); ++i) {
        //std::cout << "i: " << i << " " << arrayInfo.getDataPtr<T>()[i] << " " << arrDiff[i] << " " << arrayInfo.getDataPtr<T>()[i] - arrDiff[i] << std::endl;
        #if !defined(PREPROCESS_CPP)
        //if (arrayInfo.getDataPtr<T>()[i] - arrDiff[i] >= eps) {
        //    std::cout << "i: " << i << " " << arrayInfo.getDataPtr<T>()[i] << " " << arrDiff[i] << " " << arrayInfo.getDataPtr<T>()[i] - arrDiff[i] << std::endl;
        //}
        assert(arrayInfo.getDataPtr<T>()[i] - arrDiff[i] < eps);
        #endif
    }
    std::cout << "[CHECK PASS] " << mess << std::endl;
}
};

#ifdef TEST_NPY_CPP
int main() {
    npy_cpp::NumpyCpp np;

    double arrDouble[] = {1.0, 2.0, 3.0, 4.0};
    float arrFloat[] = {5.0, 6.0, 7.0, 8.0};
    int64_t arrInt64[] = {10, 6, 0xffffffff12, 0xffffffffffffff};
    int arrInt[] = {10, 6, 92, 82};
    const int8_t arrInt8[] = {0, 1, 0, 1};
    char arrChar[] = {'a', 'b', 'c', 'd'};
    int N = 10;
    float ans = 100.23;

    np.insert("arrDouble", (arrDouble), sizeof(arrDouble));
    np.insert("arrFloat", (arrFloat), sizeof(arrFloat));
    np.insert("arrInt64", (arrInt64), sizeof(arrInt64));
    np.insert("arrInt", (arrInt), sizeof(arrInt));
    np.insert("arrInt8", (arrInt8), sizeof(arrInt8));
    // np.insert("arrInt8", const_cast<int8_t*>(arrInt8), sizeof(arrInt8));
    np.insert("arrChar", (arrChar), sizeof(arrChar));
    np.insert("N", (&N), sizeof(N));
    np.insert("ans", (&ans), sizeof(ans));

    np.savez("file.txt");
    np.loadz("file.txt");

    npy_cpp::printArray<double>(np["arrDouble"], "arrDouble");
    npy_cpp::printArray<float>(np["arrFloat"], "arrFloat");
    npy_cpp::printArray<int64_t>(np["arrInt64"], "arrInt64");
    npy_cpp::printArray<int>(np["arrInt"], "arrInt");
    npy_cpp::printArray<int8_t>(np["arrInt8"], "arrInt8");
    npy_cpp::printArray<char>(np["arrChar"], "arrChar");
    npy_cpp::printArray<int>(np["N"], "N");
    npy_cpp::printArray<float>(np["ans"], "ans");

    npy_cpp::check(np["arrDouble"], arrDouble, sizeof(arrDouble), "arrDouble");
    npy_cpp::check(np["arrFloat"], arrFloat, sizeof(arrFloat), "arrFloat");
    npy_cpp::check(np["arrInt64"], arrInt64, sizeof(arrInt64), "arrInt64");
    npy_cpp::check(np["arrInt"], arrInt, sizeof(arrInt), "arrInt");
    npy_cpp::check(np["arrInt8"], arrInt8, sizeof(arrInt8), "arrInt8");
    npy_cpp::check(np["arrChar"], arrChar, sizeof(arrChar), "arrChar");
    npy_cpp::check(np["N"], &N, sizeof(N), "N");
    npy_cpp::check(np["ans"], &ans, sizeof(ans), "ans");
    return 0;
}
#endif
