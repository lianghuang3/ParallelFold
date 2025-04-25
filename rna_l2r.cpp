// g++ --std=c++11 rna_mp.cpp -o rna_mp -fopenmp
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>
#include <omp.h> // Include OpenMP header
#include <tuple>

// Function to count RNA secondary structures using the KT formulation, parallelized over i for each span.
long long total0(std::string s, bool verbose) {
    int n = s.size();

    s = " " + s; // now 1-based

    // Allowed pairs as two-character strings.
    std::unordered_set<std::string> allowed = {"AU", "UA", "CG", "GC", "GU", "UG"};

    // Create a 2D DP table with dimensions (n+1) x (n+1), initialized to 0.
    std::vector<std::vector<long long>> dp(n + 1, std::vector<long long>(n + 1, 0));

    // follow LP pseudocode:
    // Base cases: dp[j][j-1] = 1 for each j=1..n (empty subsequence)    
    for (int j = 1; j <= n; ++j) 
        dp[j][j-1] = 1;

    // left to right
    for (int j = 1; j <= n; j++) {
        // Parallelize the loop over i for the current span.

        for (int i = 1; i <= j; ++i) {            
            // Case 1: skip (j is unpaired)
            dp[i][j] = dp[i][j-1];

            // Case 2: {i-1} pair with j? 
            std::string pair = {s[i-1], s[j]};
            if (allowed.count(pair)) {    
                for (int k = 1; k <= i-1; ++k) {  // k<=i-1 not k<=i-2 !
                    dp[k][j] += dp[k][i-2] * dp[i][j-1]; //single-element write lock
                }
            }
        }
        if (verbose) {
            std::cout << "j=" << j << std::endl;
            for (int i = 1; i <= j; ++i) 
                std::cout << i<<"-"<<j<<":"<< dp[i][j] << " ";
            std::cout << std::endl;
        }
    }
    return dp[1][n];
}


// atomic update
long long total1(std::string s, bool verbose) {
    int n = s.size();

    s = " " + s; // now 1-based

    // Allowed pairs as two-character strings.
    std::unordered_set<std::string> allowed = {"AU", "UA", "CG", "GC", "GU", "UG"};

    // Create a 2D DP table with dimensions (n+1) x (n+1), initialized to 0.
    std::vector<std::vector<long long>> dp(n + 1, std::vector<long long>(n + 1, 0));

    // follow LP pseudocode:
    // Base cases: dp[j][j-1] = 1 for each j=1..n (empty subsequence)    
    for (int j = 1; j <= n; ++j) 
        dp[j][j-1] = 1;

    // left to right
    for (int j = 1; j <= n; j++) {
        // Parallelize the loop over i for the current span.

        #pragma omp parallel for schedule(dynamic)

        for (int i = 1; i <= j; ++i) {            
            // Case 1: skip (j is unpaired)
            dp[i][j] = dp[i][j-1];
        }

        #pragma omp parallel for schedule(dynamic)

        for (int i = 1; i <= j; ++i) {            

            // Case 2: {i-1} pair with j? 
            std::string pair = {s[i-1], s[j]};
            if (allowed.count(pair)) {    

                #pragma omp parallel for schedule(dynamic) // it's faster not having this parallel; but why?

                for (int k = 1; k <= i-1; ++k) {  // k<=i-1 not k<=i-2 !

                    long long a = dp[k][i-2] * dp[i][j-1];

                    #pragma omp atomic update
                    dp[k][j] += a; //single-element write lock
                }
            }
        }
        if (verbose) {
            std::cout << "j=" << j << std::endl;
            for (int i = 1; i <= j; ++i) 
                std::cout << i<<"-"<<j<<":"<< dp[i][j] << " ";
            std::cout << std::endl;
        }
    }
    return dp[1][n];
}


// throw to 2D array (vector), then collect
long long total2(std::string s, bool verbose) {
    int n = s.size();

    s = " " + s; // now 1-based

    // Allowed pairs as two-character strings.
    std::unordered_set<std::string> allowed = {"AU", "UA", "CG", "GC", "GU", "UG"};

    // Create a 2D DP table with dimensions (n+1) x (n+1), initialized to 0.
    std::vector<std::vector<long long>> dp(n + 1, std::vector<long long>(n + 1, 0));

    // follow LP pseudocode:
    // Base cases: dp[j][j-1] = 1 for each j=1..n (empty subsequence)    
    for (int j = 1; j <= n; ++j) 
        dp[j][j-1] = 1;

    // this line has to be outside; otherwise mem alloc is too slow  
    std::vector<std::vector<long long>> dp_tmp(n + 1, std::vector<long long>(n + 1, 0));

    // left to right
    for (int j = 1; j <= n; j++) {
        // Parallelize the loop over i for the current span.

        //dp_tmp.clear(); // runtime error
        // #pragma omp parallel for schedule(dynamic), collapse(2)
        // for (int i = 1; i <= j; ++i)
        //     for (int k=1; k <= i-1; ++k)
        //         dp_tmp[k][i] = 0; 


        #pragma omp parallel for schedule(dynamic)

        for (int i = 1; i <= j; ++i) {            
            // Case 1: skip (j is unpaired)
            dp[i][j] = dp[i][j-1];
        }

        
        #pragma omp parallel for schedule(dynamic)

        for (int i = 1; i <= j; ++i) {            

            // Case 2: {i-1} pair with j? 
            std::string pair = {s[i-1], s[j]};
            if (allowed.count(pair)) {    

                //#pragma omp parallel for schedule(dynamic) // doesn't matter whether this line or not

                for (int k = 1; k <= i-1; ++k) {  // k<=i-1 not k<=i-2 !

                    dp_tmp[k][i] = dp[k][i-2] * dp[i][j-1]; //single-element write lock
                }
            }
            else { // can be faster!
                #pragma omp parallel for schedule(dynamic)                
                for (int k=1; k<=i-1; ++k)
                    dp_tmp[k][i] = 0;
            }

        }

        #pragma omp parallel for schedule(dynamic) // important
        // parallel sum
        for (int k = 1; k <= j; ++k) {       
            long long a = 0;
            #pragma omp parallel for shared(k,j),reduction(+:a) // seems no use
            for (int i=k+1; i<=j;i++) 
                //if (allowed.count({s[i-1], s[j]})) // too slow
                a += dp_tmp[k][i];
            dp[k][j] += a;
        }

        if (verbose) {
            std::cout << "j=" << j << std::endl;
            for (int i = 1; i <= j; ++i) 
                std::cout << i<<"-"<<j<<":"<< dp[i][j] << " ";
            std::cout << std::endl;
        }
    }
    return dp[1][n];
}

// throw to hash w/ random key, then collect
long long total3(std::string s, bool verbose) {
    int n = s.size();

    s = " " + s; // now 1-based

    // Allowed pairs as two-character strings.
    std::unordered_set<std::string> allowed = {"AU", "UA", "CG", "GC", "GU", "UG"};

    // Create a 2D DP table with dimensions (n+1) x (n+1), initialized to 0.
    std::vector<std::vector<long long>> dp(n + 1, std::vector<long long>(n + 1, 0));

    // follow LP pseudocode:
    // Base cases: dp[j][j-1] = 1 for each j=1..n (empty subsequence)    
    for (int j = 1; j <= n; ++j) 
        dp[j][j-1] = 1;

    // this line has to be outside; otherwise mem alloc is too slow  
    // from random-key to (k, value)
    std::unordered_map<long, std::pair<int, long long>> dp_tmp;

    // left to right
    for (int j = 1; j <= n; j++) {
        // Parallelize the loop over i for the current span.

        //dp_tmp.clear(); // runtime error
        // #pragma omp parallel for schedule(dynamic), collapse(2)
        // for (int i = 1; i <= j; ++i)
        //     for (int k=1; k <= i-1; ++k)
        //         dp_tmp[k][i] = 0; 

        dp_tmp.clear();

        #pragma omp parallel for schedule(dynamic)

        for (int i = 1; i <= j; ++i) {            
            // Case 1: skip (j is unpaired)
            dp[i][j] = dp[i][j-1];
        }

        
        #pragma omp parallel for schedule(dynamic)

        for (int i = 1; i <= j; ++i) {            

            // Case 2: {i-1} pair with j? 
            std::string pair = {s[i-1], s[j]};
            if (allowed.count(pair)) {    

                //#pragma omp parallel for schedule(dynamic) // doesn't matter whether this line or not

                for (int k = 1; k <= i-1; ++k) {  // k<=i-1 not k<=i-2 !
                    // very slow!
                    long long a =  dp[k][i-2] * dp[i][j-1];
                    //std::cout << k << " " << i << " " << a << std::endl;

                    dp_tmp[k*n + i] = std::make_pair(k, a); //single-element write lock
                }
            }
            // else { // can be faster!
            //     #pragma omp parallel for schedule(dynamic)                
            //     for (int k=1; k<=i-1; ++k)
            //         dp_tmp[k][i] = 0;
            // }

        }

        //#pragma omp parallel for // important
        // doesn't compile!
        if (false) for (auto & kv: dp_tmp) {
            int i = kv.first % n;
            int k = kv.second.first;
            long long a = kv.second.second;

            //#pragma omp atomic update
            dp[k][j] += a;
        }

        if (verbose) {
            std::cout << "j=" << j << std::endl;
            for (int i = 1; i <= j; ++i) 
                std::cout << i<<"-"<<j<<":"<< dp[i][j] << " ";
            std::cout << std::endl;
        }
    }
    return dp[1][n];
}

int main() {
    std::string test0 = "ACAGU";
    std::string test10 = "AUUCUGGUUG";
    // n=100
    std::string test100 = "AUUCUGGUUGAUCCUGCCAGAGGCCGCUGCUAUCCGGCUGGGACUAAGCCAUGCGAGUCAAGGGGCUUGUAUCCCUUCGGGGAUGCAAGCACCGGCGGAC";
    // Test cases: 16S rRNA    
    std::string test1 = "AUUCUGGUUGAUCCUGCCAGAGGCCGCUGCUAUCCGGCUGGGACUAAGCCAUGCGAGUCAAGGGGCUUGUAUCCCUUCGGGGAUGCAAGCACCGGCGGACGGCUCAGUAACACGUGGACAACCUGCCCUCGGGUGGGGGAUAACCCCGGGAAACUGGGGCUAAUCCCCCAUAGGGGAUGGGUACUGGAAUGUCCCAUCUCCGAAAGCGCUUAGCGCCCGAGGAUGGGUCUGCGGCGGAUUAGGUUGUUGGUGGGGUAACGGCCCACCAAGCCGAAGAUCCGUACGGGCCAUGAGAGUGGGAGCCCGGAGAUGGACCCUGAGACACGGGUCCAGGCCCUACGGGGCGCAGCAGGCGCGAAACCUCCGCAAUGCGGGAAACCGCGACGGGGUCAGCCGGAGUGCUCGCGCAUCGCGCGGGCUGUCGGGGUGCCUAAAAAGCACCCCACAGCAAGGGCCGGGCAAGGCCGGUGGCAGCCGCCGCGGUAAUACCGGCGGCCCGAGUGGCGGCCACUUUUAUUGGGCCUAAAGCGUCCGUAGCCGGGCUGGUAAGUCCUCCGGGAAAUCUGGCGGCUUAACCGUCAGACUGCCGGAGGAUACUGCCAGCCUAGGGACCGGGAGAGGCCGGGGGUAUUCCCGGAGUAGGGGUGAAAUCCUGUAAUCCCGGGAGGACCACCUGUGGCGAAGGCGCCCGGCUGGAACGGGUCCGACGGUGAGGGACGAAGGCCAGGGGAGCGAACCGGAUUAGAUACCCGGGUAGUCCUGGCUGUAAACGAUGCGGACUAGGUGUCACCGAAGCUACGAGCUUCGGUGGUGCCGGAGGGAAGCCGUUAAGUCCGCCGCCUGGGGAGUACGGCCGCAAGGCUGAAACUUAAAGGAAUUGGCGGGGGAGCACUACAACGGGUGGAGCCUGCGGUUUAAUUGGAUUCAACGCCGGGAAGCUUACCGGGGGAGACAGCGGGAUGAAGGUCGGGCUGAAGACCUUACCAGACUAGCUGAGAGGUGGUGCAUGGCCGCCGUCAGUUCGUACUGUGAAGCAUCCUGUUAAGUCAGGCAACGAGCGAGACCCGCGCCCCCAGUUGCCAGCGGUUCCCUUCGGGGAAGCCGGGCACACUGGGGGGACUGCCGGCGCUAAGCCGGAGGAAGGUGCGGGCAACGGCAGGUCCGUAUGCCCCGAAUCCCCCGGGCUACACGCGGGCUACAAUGGCCGGGACAAUGGGUACCGACCCCGAAAGGGGUAGGUAAUCCCCUAAACCCGGUCUAACCUGGGAUCGAGGGCUGCAACUCGCCCUCGUGAACCUGGAAUCCGUAGUAAUCGCGCCUCAAAAUGGCGCGGUGAAUACGUCCCUGCUCCUUGCACACACCGCCCGUCAAGCCACCCGAGUGGGCCAGGGGCGAGGGGGUGGCCCUAGGCCACCUUCGAGCCCAGGGUCCGCGAGGGGGGCUAAGUCGUAACAAGGUAGCCGUAGGGGAAUCUGCGGCUGGAUCACCUCCU";
    std::string test2 = "UUCCCUGAAGAGUUUGAUCCUGGCUCAGCGCGAACGCUGGCGGCGUGCCUAACACAUGCAAGUCGUGCGCAGGCUCGCUCCCUCUGGGAGCGGGUGCUGAGCGGCAAACGGGUGAGUAACACGUGGGUAACCUACCCCCAGGAGGGGGAUAACCCCGGGAAACCGGGGCUAAUACCCCAUAAAGCCGCCCGCCACUAAGGCGAGGCGGCCAAAGGGGGCCUCUGGGCUCUGCCCAAGCUCCCGCCUGGGGAUGGGCCCGCGGCCCAUCAGGUAGUUGGUGGGGUAACGGCCCACCAAGCCUAUGACGGGUAGCCGGCCUGAGAGGGUGGCCGGCCACAGCGGGACUGAGACACGGCCCGCACCCCUACGGGGGGCAGCAGUGGGGAAUCGUGGGCAAUGGGCGAAAGCCUGACCCCGCGACGCCGCGUGGGGGAAGAAGCCCUGCGGGGUGUAAACCCCUGUCGGGGGGGACGAAGGGACUGUGGGUUAAUAGCCCACAGUCUUGACGGUACCCCCAGAGGAAGGGACGGCUAACUACGUGCCAGCAGCCGCGGUAAUACGUAGGUCCCGAGCGUUGCGCGAAGUCACUGGGCGUAAAGCGUCCGCAGCCGGUCGGGUAAGCGGGAUGUCAAAGCCCACGGCUCAACCGUGGAAUGGCAUCCCGAACUGCCCGACUUGAGGCACGCCCGGGCAGGCGGAAUUCCCGGGGUAGCGGUGAAAUGCGUAGAUCUCGGGAGGAACACCGAAGGGGAAGCCAGCCUGCUGGGGCUGUCCUGACGGUCAGGGACGAAAGCCGGGGGAGCGAACCGGAUUAGAUACCCGGGUAGUCCCGGCCGUAAACCAUGGGCGCUAGGGCUUGUCCCUUUGGGGCAGGCUCGCAGCUAACGCGUUAAGCGCCCCGCCUGGGGAGUACGGGCGCAAGCCUGAAACUCAAAGGAAUUGGCGGGGGCCCGCACAACCGGUGGAGCGUCUGGUUCAAUUCGAUGCUAACCGAAGAACCUUACCCGGGCUUGACAUGCCGGGGAGACUCCGCGAAAGCGGAGUUGUGGAAGUCUCUGACUUCCCCCCGGCACAGGUGGUGCAUGGCCGUCGUCAGCUCGUGUCGUGAGAUGUUGGGUUAAGUCCCGCAACGAGCGCAACCCCUGCCCCUAGUUGCUACCCCGAGAGGGGAGCACUCUAGGGGGACCGCCGGCGAUAAGCCGGAGGAAGGGGGGGAUGACGUCAGGUCAGUAUGCCCUUUAUGCCCGGGGCCACACAGGCGCUACAGUGGCCGGGACAAUGGGAAGCGACCCCGCAAGGGGGAGCUAAUCCCAGAAACCCGGUCAUGGUGCGGAUUGGGGGCUGAAACUCGCCCCCAUGAAGCCGGAAUCGGUAGUAACGGGGUAUCAGCGAUGUCCCCGUGAAUACGUUCUCGGGCCUUGCACACACCGCCCGUCACGCCACGGAAGUCGGUCCGGCCGGAAGUCCCCGAGCUAACCGGCCCUUUUUGGGCCGGGGGCAGGGGCCGAUGGCCGGGCCGGCGACUGGGGCGAAGUCGUAACAAGGUAGCCGUAGGGGAACCUGC";
    // Test Cases: GRP II 
    std::string test3 = "AGUUUAGUGGUAAAAGUGUGAUUCGUUCUAUUAUCCCUUAAAUAGUUAAAGGGUCCUUCGGUUUGAUUCGUAUUCCGAUCAAAAACUUGAUUUCUAAAAAGGAUUUAAUCCUUUUCCUCUCAAUGACAGAUUCGAGAACAAAUACACAUUCUCGUGAUUUGUAUCCAAGGGUCACUUAGACAUUGAAAAAUUGGAUUAUGAAAUUGCGAAACAUAAUUUUGGAAUUGGAUCAAUACUUCCAAUUGAAUAAGUAUGAAUAAAGGAUCCAUGGAUGAAGAUAGAAAGUUGAUUUCUAAUCGUAACUAAAUCUUCAAUUUCUUAUUUGUAAAGAAGAAAUUGAAGCAAAAUAGCUAUUAAACGAUGACUUUGGUUUACUAGAGACAUCAACAUAUUGUUUUAGCUCGGUGGAAACAAAACCCUUUUCCUCAGGAUCCUAUUAAAUAGAAAUAGAGAACGAAAUAACUAGAAAGGUUGUUAGAAUCCCCUCUUCUAGAAGGAUCAUCUACAAAGCUAUUCGUUUUAUCUGUAUUCAGACCAAAAGCUGACAUAGAUGUUAUGGGUAGAAUUCUUUUUUUUUUUCGAAUUUUGUUCACAUCUUAGAUCUAUAAAUUGACUCAUCUCCAUAAAGGAGCCGAAUGAAACCAAAGUUUCAUGUUCGGUUUUGAAUUAGAGACGUUAAAAAUAAUGAAUCGACGUCGACUAUAACCC";
    std::string test4 = "GCUAGGGAUAACAGGGUGCGACCUGCCAAGCUGCACAAUUCAAUGUGGUUAGAAAACCAACUUGGAAUCCAAUCUCCAUGAGCCUACCAUCACAUGCGUUCUAGGGUUAACCUGAAGGUGUGAAGCUGAUGGGAAAAAGUAACCCAAACUGUAUGUGACAGUGAGGGGGCAGGCUAGAUUCCUAUGGGCAAUGUAAAUGAACACUCAUCUGAGGCAUCGUGACCCUAUCACAUCUAGUUAAUUGUGAGAGAAUCUUAUGUCUCUGUUUCAUAAGAUUGAUUGGACAAUUUCUCACCGGUGUAAAGAGUGGUCCUAAGGGAAUCAUCGAAAGUGAAUUGUGCGGAACAGGGCAAGCCCCAUAGGCUCCUUCGGGAGUGAGCGAAGCAAUUCUCUCUAUCGCCUAGUGGGUAAAAGACAGGGCAAAAAGCGAAUGCCAGUUAGUAAUAGACUGGAUAGGGUGAAUAACCUAACCUGAAAGGGUGCAGACUUGCUCAUGGGCGUGGGAAAUCAGAUUUCGUGAAUACACCAGCAUUCAAGAGUUUCCAUGCUUGAGCCGUGUGCGGUGAAAGUCGCAUGCACGGUUCUACUGGGGGGAAAGCCUGAGAGGGCCUACCUAUCCAACUUU";
    // std::string test5 = "AGGCAUCAAACCCUGCAUGGGAGCGGAUGCCGUGUAGUCCAAAGACUUCUUUGGCACUAAGGCAUCAAACCCUGCAUGGGAGCGGAUGCCGUGUAGUCCAAAGACUUCUUUGGCACUAAGGCAUCAAACCCUGCAUGGGAGCGGAUGCCGUGUAGUCCAAAGACUUCUUUGGCACUAAGGCAUCAAACCCUGCAUGGGAGCGGAUGCCGUGUAGUCCAAAGACUUCUUUGGCACUAAGGCAUCAAACCCUGCAUGGGAGCGGAUGCCGUGUAGUCCAAAGACUUCUUUGGCACUAAGGCAUCAAACCCUGCAUGGGAGCGGAUGCCGUGUAGUCCAAAGACUUCUUUGGCACUAAGGCAUCAAACCCUGCAUGGGAGCGGAUGCCGUGUAGUCCAAAGACUUCUUUGGCACUAAGGCAUCAAACCCUGCAUGGGAGCGGAUGCCGUGUAGUCCAAAGACUUCUUUGGCACUA";

    // test2 (1.5k): on 8x: serial 2.07s; total1: 2.96s; total2: 1.25s
    std::string test5 = test2 + test2; // 3k;  on 8x: serial 18.8s; total1: 22s; total2: 7.7s (limit: 6.1s)
    std::string test6 = test5 + test2; // 4.5k; on 8x: serial 165s; total1: 52s; total2: 44s (limit: 30.8s)

    //std::cout << "total(\"" << test0 << "\") = " << total0(test0, true) << "\n";
    //std::cout << "total(\"" << test0 << "\") = " << total1(test0, true) << "\n";
    //std::cout << "total(\"" << test2 << "\") = " << total0(test2, false) << "\n";
    std::cout << "total(\"" << test0 << "\") = " << total0(test2, false) << "\n";
    //std::cout << "total(\"" << test1 << "\") = " << total(test1, false) << "\n";
    //std::cout << "total(\"" << test4 << "\") = " << total(test4, false) << "\n";

    return 0;
}
