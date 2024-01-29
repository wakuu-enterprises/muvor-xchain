// macox
// g++ -dynamiclib -flat_namespace myclass.cc -o myclass.so
// g++ class_user.cc -o class_user
// linux
// g++ -fPIC -shared myclass.cc -o myclass.so
// g++ class_user.cc -ldl -o class_user
#include <libsnark/gadgetlib1/gadget.hpp>
#include <libsnark/gadgetlib1/gadgets/hashes/sha256/sha256_gadget.hpp>
#include <libsnark/common/default_types/r1cs_ppzksnark_pp.hpp>

extern "C"
{
    using namespace libsnark;

    typedef default_r1cs_ppzksnark_pp ppT;
    typedef r1cs_ppzksnark_keypair<ppT> keypairT;
    typedef r1cs_ppzksnark_proof<ppT> proofT;
    typedef r1cs_ppzksnark_primary_input<ppT> primary_inputT;
    typedef r1cs_ppzksnark_auxiliary_input<ppT> auxiliary_inputT;

    keypairT *zk_snark_generate_keypair(char *pk_path, char *vk_path)
    {
        return new keypairT(keypairT::generate(ppT::get_default_pp(), true));
    }

    proofT *zk_snark_prove(keypairT *keypair, primary_inputT *primary_input, auxiliary_inputT *auxiliary_input)
    {
        return new proofT(keypair->prover(proofT::primary_input_type(*primary_input), proofT::auxiliary_input_type(*auxiliary_input)));
    }

    int zk_snark_verify(keypairT *keypair, proofT *proof, primary_inputT *primary_input)
    {
        return keypair->verifier(proofT::primary_input_type(*primary_input), *proof);
    }
}
