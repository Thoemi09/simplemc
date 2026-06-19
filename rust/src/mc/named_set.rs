use crate::{Result, RmcError};

pub trait NamedEntry {
    fn name(&self) -> &str;
}

#[derive(Clone, Debug)]
pub struct NamedSet<T> {
    entries: Vec<T>,
}

impl<T> Default for NamedSet<T> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl<T> NamedSet<T>
where
    T: NamedEntry,
{
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn add(&mut self, entry: T, kind: &str) -> Result<()> {
        if self.find(entry.name()).is_some() {
            return Err(RmcError::InvalidArgument(format!(
                "{} '{}' is already registered",
                kind,
                entry.name()
            )));
        }
        self.entries.push(entry);
        Ok(())
    }

    pub fn find(&self, name: &str) -> Option<usize> {
        self.entries.iter().position(|entry| entry.name() == name)
    }

    pub fn get_by_name(&self, name: &str) -> Option<&T> {
        self.find(name).map(|idx| &self.entries[idx])
    }

    pub fn get_by_name_mut(&mut self, name: &str) -> Option<&mut T> {
        self.find(name).map(|idx| &mut self.entries[idx])
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[T] {
        &self.entries
    }

    pub fn entries_mut(&mut self) -> &mut [T] {
        &mut self.entries
    }
}

impl<T> std::ops::Index<usize> for NamedSet<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.entries[index]
    }
}

impl<T> std::ops::IndexMut<usize> for NamedSet<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.entries[index]
    }
}
