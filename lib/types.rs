use bias_core::cio::{TypeDB, TypeError, TypeInfoDB};
use bias_core::ir::{Term, Type, TypeKind};
use hashbrown::HashMap;
use thiserror::Error;

#[derive(Debug, Clone)]
#[allow(unused)]
pub struct TypeFactory {
    /// Backing TypeDB that holds all types we could potentially infer
    type_db: TypeDB,
    /// Alternative representation of backing types
    typeinfo_db: TypeInfoDB,
    /// Cache to minimise full type constructions
    added_types: HashMap<String, Term<Type>>,
}

#[derive(Error, Debug)]
pub enum TypeFactoryError {
    #[error("failed to load type database: {0}")]
    DbLoad(#[from] TypeError),
    #[error("failed to resolve type '{name}': {kind}")]
    TypeResolution {
        name: String,
        kind: TypeResolutionError,
    },
}

#[derive(Error, Debug)]
pub enum TypeResolutionError {
    #[error("type not found in database")]
    NotFound,
    #[error("unsupported bit width: {0}")]
    InvalidBitWidth(u32),
    #[error("invalid type format")]
    InvalidFormat,
}

impl TypeFactory {
    pub fn new(typeinfo_db: TypeInfoDB) -> Self {
        let mut new_typedb = TypeDB::new();
        new_typedb.import_types(&typeinfo_db);
        Self {
            typeinfo_db: typeinfo_db.to_owned(),
            type_db: new_typedb,
            added_types: HashMap::new(),
        }
    }

    pub fn from_file(path: &str) -> Result<Self, TypeFactoryError> {
        let typeinfo_db = TypeInfoDB::from_file(path)?;
        let mut new_typedb = TypeDB::new();
        new_typedb.import_types(&typeinfo_db);
        Ok(Self {
            type_db: new_typedb,
            typeinfo_db,
            added_types: HashMap::new(),
        })
    }

    pub(crate) fn type_info(&self) -> &TypeInfoDB {
        &self.typeinfo_db
    }

    pub(crate) fn type_db(&self) -> &TypeDB {
        &self.type_db
    }
}

pub fn is_primitive(typ: &Term<Type>) -> bool {
    match typ.kind() {
        TypeKind::Bool => true,
        TypeKind::Unsigned(_) => true,
        TypeKind::Signed(_) => true,
        TypeKind::Pointer(inner, _) => match inner.kind() {
            TypeKind::Unsigned(_) => !typ.is_unsigned_char(),
            TypeKind::Signed(_) => !typ.is_char(),
            _ => false,
        },
        _ => false,
    }
}
