#ifndef __HAMILTONIAN_H__
#define __HAMILTONIAN_H__

#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <complex>
#include <stdexcept>
#include <cstdint>
#include <iomanip>      // std::setprecision
#include <algorithm>

#include "switch_backend.h"


// Hamiltonian
// load from hamiltonian file and preprocess
namespace Ham
{
    using namespace std::complex_literals;

    using datatype = int8_t;
    using coeff_dtype = double;
    using BigIntType = std::vector<uint64_t>;
    // using QubitOpType = std::map<std::vector<std::pair<int, char>>, std::complex<double>>;
    using QubitOpType = std::vector<std::pair<std::vector<std::pair<int, char>>, std::complex<double>>>;

    template<typename T>
    void dump_vector(std::vector<T> vec) {
        for (int i = 0; i < vec.size(); i++) {
            std::cout << vec[i] << " ";
        }
        std::cout << std::endl;
    }

    class Hamiltonian
    {
        public:
            Hamiltonian() : nQubits_(-1) {}
            Hamiltonian(int nQubits) : nQubits_(nQubits) {}
            Hamiltonian(std::string fn) {
                getHamiltonian(fn);
            }

            void getHamiltonian(std::string fn) {
                auto qubit_op = read_binary_qubit_op(fn);
                // dumpQubitOp(qubit_op);
                // return;
                extractIndices(qubit_op, 0.0);
                // dump();
            }

            void transferToC() {
                set_indices_ham_int_opt(
                    nQubits_,
                    K_,
                    NK_,
                    idxs_.data(),
                    coeffsBuf_.data(),
                    pauliMat12Buf_.data(),
                    pauliMat23Buf_.data());
            }

            void extractIndices(const QubitOpType &qubitOps, double eps);

            int nQubits() const { return nQubits_; }

            QubitOpType read_binary_qubit_op(const std::string& filename);

            void dumpQubitOp(const QubitOpType &qubitOp);

            void dump() {
                std::cout << "nQubits: " << nQubits_ << " K: " << K_ << " NK: " << NK_ << std::endl;
                // std::cout << "PauliMat12Buf===: " << pauliMat12Buf_.size() << std::endl;
                for (int i = 0; i < pauliMat12Buf_.size()/nQubits_; i++) {
                    for (int j = 0; j < nQubits_; j++) {
                        std::cout << (int)pauliMat12Buf_[i*nQubits_+j] << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;

                // std::cout << "PauliMat23Buf===: " << pauliMat23Buf_.size() << " qubits: " << nQubits_ << std::endl;
                for (int i = 0; i < pauliMat23Buf_.size()/nQubits_; i++) {
                    for (int j = 0; j < nQubits_; j++) {
                        std::cout << (int)pauliMat23Buf_[i*nQubits_+j] << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;

                for (int i = 0; i < idxs_.size(); i++) {
                    std::cout << idxs_[i] << "\n";
                }
                std::cout << std::endl;
                // std::cout << "coeffsBuf_===: " << coeffsBuf_.size() << std::endl;
                // std::sort(coeffsBuf_.begin(), coeffsBuf_.end());
                for (int i = 0; i < coeffsBuf_.size(); i++) {
                    std::cout << std::setprecision(15) << coeffsBuf_[i] << "\n";
                }
                std::cout << std::endl;

                // std::cout << "idxs: "<< idxs_.size() << std::endl;
            }

        private:
            int nQubits_;

            std::vector<datatype> pauliMat12Buf_;
            std::vector<datatype> pauliMat23Buf_;
            std::vector<coeff_dtype> coeffsBuf_;
            std::vector<int64_t> idxs_;
            int64_t K_, NK_;

            BigIntType state2Id(const std::vector<datatype> &state);
            double real_pow_neg_1i(int cnt) {
                static const std::array<double, 4> lookup = {1.0, 0.0, -1.0, 0.0};
                return lookup[cnt % 4];
            }
    };

    BigIntType Hamiltonian::state2Id(const std::vector<datatype> &state) {
        BigIntType res;

        for (int i = 0; i < state.size(); i += 64) {
            uint64_t bitValue = 1;
            uint64_t val = 0;
            for (int j = 0; j < 64 && i + j < state.size(); j++) {
                if (state[i + j] == 1) {
                    val += bitValue;
                }
                bitValue *= 2;
            }
            res.push_back(val);
        }

        return res;
    }

    void Hamiltonian::extractIndices(const QubitOpType &qubitOps, double eps) {
        K_ = qubitOps.size();

        std::map<BigIntType, std::vector<datatype>> pauliMat12Dict_;
        std::map<BigIntType, std::vector<std::vector<datatype>>> pauliMat23Dict_;
        std::map<BigIntType, std::vector<coeff_dtype>> coeffsDict_;

        for (const auto &term : qubitOps) {
            int cnt = 0;
            std::vector<datatype> pauliMat12(nQubits_, 0);
            std::vector<datatype> pauliMat23(nQubits_, 0);
            // for (const auto &[pos, pauli] : term.first) {
            auto pauli_str = term.first;
            for (const auto &[pos, pauli] : pauli_str) {
                if (pauli == 'X') {
                    pauliMat12[pos] = 1;
                } else if (pauli == 'Y') {
                    pauliMat12[pos] = 1;
                    pauliMat23[pos] = 1;
                    cnt++;
                } else {
                    pauliMat23[pos] = 1;
                }
            }

            // auto coeff = std::real(term.second) * std::real(std::pow(-1.0i, cnt));
            auto coeff = std::real(term.second) * real_pow_neg_1i(cnt);
            // std::cout << "coeff: " << coeff << std::endl;
            // dump_vector(pauliMat12);
            auto stateId = state2Id(pauliMat12);
            // dump_vector(stateId);
            if (coeffsDict_.count(stateId)) {
                coeffsDict_[stateId].push_back(coeff);
                pauliMat23Dict_[stateId].push_back(pauliMat23);
            } else {
                pauliMat12Dict_[stateId] = pauliMat12;
                pauliMat23Dict_[stateId].push_back(pauliMat23);
                coeffsDict_[stateId] = {coeff};
            }
        }

        NK_  = pauliMat12Dict_.size();
        // Assign to continuous buffers
        int numIdx = 0;
        std::vector<int> num_list;
        idxs_.push_back(0);
        for (auto &[sId, pMat23] : pauliMat23Dict_) {
            // dump_vector(coeffsDict_[sId]);
            coeffsBuf_.insert(coeffsBuf_.end(), coeffsDict_[sId].begin(), coeffsDict_[sId].end());
            pauliMat12Buf_.insert(pauliMat12Buf_.end(), pauliMat12Dict_[sId].begin(), pauliMat12Dict_[sId].end());
            int num = pMat23.size();
            for (int j = 0; j < num; j++) {
                pauliMat23Buf_.insert(pauliMat23Buf_.end(), pMat23[j].begin(), pMat23[j].end());
            }
            numIdx += num;
            num_list.push_back(num);
            idxs_.push_back(numIdx);
        }

        // sort(num_list.begin(), num_list.end());
        //for (int i : num_list) {
        //    printf("%d\n", i);
        //}
    }

    QubitOpType Hamiltonian::read_binary_qubit_op(const std::string& filename) {
        const double magic_number = 11.2552;
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Unable to open file");
        }

        double identifier;
        file.read(reinterpret_cast<char*>(&identifier), sizeof(identifier));
        if (identifier != magic_number) {
            throw std::runtime_error("The file is not saved by QCQC.");
        }

        int32_t n_qubits;
        file.read(reinterpret_cast<char*>(&n_qubits), sizeof(n_qubits));
        nQubits_ = n_qubits;

        std::map<char, char> pauli_symbol_dict{
            {0, 'I'},
            {1, 'X'},
            {2, 'Y'},
            {3, 'Z'}
        };

        QubitOpType qubit_op_dict;

        while (file) {
            std::complex<double> coeff;
            file.read(reinterpret_cast<char*>(&coeff), sizeof(coeff));

            std::vector<int32_t> pauli_str_tmp(n_qubits);
            file.read(reinterpret_cast<char*>(pauli_str_tmp.data()), sizeof(int32_t) * n_qubits);

            if (file) {
                std::vector<std::pair<int, char>> pauli_str_tuple;
                for (int i = 0; i < n_qubits; ++i) {
                    if (pauli_str_tmp[i] != 0) {
                        pauli_str_tuple.emplace_back(i, pauli_symbol_dict[pauli_str_tmp[i]]);
                    }
                }
                // qubit_op_dict[pauli_str_tuple] = coeff;
                qubit_op_dict.push_back({pauli_str_tuple, coeff});
            }
        }

        return qubit_op_dict;
    }

    void Hamiltonian::dumpQubitOp(const QubitOpType &qubitOp) {
        // Print out the qubit operations for testing purposes
        for (const auto& item : qubitOp) {
            std::cout << std::setprecision(15) << item.second.real() << std::endl;
        }
        for (const auto& item : qubitOp) {
            for (const auto& p : item.first) {
                std::cout << "(" << p.first << ", " << p.second << ") ";
            }
            std::cout << '\n';
        }
    }

    void get_preprocess_ham(std::string fn) {
        try {
            Ham::Hamiltonian ham(fn);
            ham.transferToC();
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << '\n';
        }
    }
} // namespace


#ifdef TEST_HAM
int main() {
    try {
        std::string fn = "/home/wyj/NeuralNetworkQuantumState/interface_qcqc/molecules/thomas/h2/qubit_op.data";
        // std::string fn = "/home/wyj/NeuralNetworkQuantumState/interface_qcqc/molecules/thomas/lih/qubit_op.data";
        // std::string fn = "/home/wyj/NeuralNetworkQuantumState/interface_qcqc/molecules/thomas/h2o/qubit_op.data";
        // Ham::Hamiltonian ham(fn);
        // Ham::Hamiltonian ham;
        // ham.getHamiltonian(fn);
        Ham::get_preprocess_ham(fn);
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << '\n';
    }
    return 0;
}
#endif

#endif //__HAMILTONIAN_H__
