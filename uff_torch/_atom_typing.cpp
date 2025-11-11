#include <boost/python.hpp>
#include <GraphMol/ForceFieldHelpers/UFF/AtomTyper.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/PeriodicTable.h>
#include <GraphMol/RDKitBase.h>
#include <RDGeneral/RDLog.h>

namespace {
std::string compute_atom_label(const RDKit::Atom &atom,
                               bool tolerate_charge_mismatch) {
  int atomic_num = atom.getAtomicNum();
  std::string atom_key = atom.getSymbol();
  if (atom_key.size() == 1) {
    atom_key += '_';
  }
  auto *table = RDKit::PeriodicTable::getTable();
  if (atomic_num) {
    const bool skip_hybridization =
        table->getDefaultValence(atomic_num) == -1 ||
        (table->getNouterElecs(atomic_num) != 1 &&
         table->getNouterElecs(atomic_num) != 7);
    if (skip_hybridization) {
      switch (atomic_num) {
        case 12:
        case 13:
        case 14:
        case 15:
        case 50:
        case 51:
        case 52:
        case 81:
        case 82:
        case 83:
        case 84:
          atom_key += '3';
          if (atom.getHybridization() != RDKit::Atom::SP3) {
            BOOST_LOG(rdWarningLog)
                << "UFFTYPER: Warning: hybridization set to SP3 for atom "
                << atom.getIdx() << std::endl;
          }
          break;
        case 80:
          atom_key += '1';
          if (atom.getHybridization() != RDKit::Atom::SP) {
            BOOST_LOG(rdWarningLog)
                << "UFFTYPER: Warning: hybridization set to SP for atom "
                << atom.getIdx() << std::endl;
          }
          break;
        default:
          switch (atom.getHybridization()) {
            case RDKit::Atom::S:
              break;
            case RDKit::Atom::SP:
              atom_key += '1';
              break;
            case RDKit::Atom::SP2:
              if ((atom.getIsAromatic() ||
                   RDKit::MolOps::atomHasConjugatedBond(&atom)) &&
                  (atomic_num == 6 || atomic_num == 7 || atomic_num == 8 ||
                   atomic_num == 16)) {
                atom_key += 'R';
              } else {
                atom_key += '2';
              }
              break;
            case RDKit::Atom::SP3:
              atom_key += '3';
              break;
            case RDKit::Atom::SP2D:
              atom_key += '4';
              break;
            case RDKit::Atom::SP3D:
              atom_key += '5';
              break;
            case RDKit::Atom::SP3D2:
              atom_key += '6';
              break;
            default:
              BOOST_LOG(rdErrorLog)
                  << "UFFTYPER: Unrecognized hybridization for atom: "
                  << atom.getIdx() << std::endl;
          }
      }
    }
  }
  RDKit::UFF::Tools::addAtomChargeFlags(&atom, atom_key,
                                        tolerate_charge_mismatch);
  return atom_key;
}
}  // namespace

namespace uff_torch {
std::string uff_atom_type(const RDKit::Atom &atom,
                          bool tolerate_charge_mismatch) {
  return compute_atom_label(atom, tolerate_charge_mismatch);
}
}  // namespace uff_torch

BOOST_PYTHON_MODULE(_atom_typing) {
  boost::python::def(
      "uff_atom_type", &uff_torch::uff_atom_type,
      (boost::python::arg("atom"),
       boost::python::arg("tolerate_charge_mismatch") = false));
}
