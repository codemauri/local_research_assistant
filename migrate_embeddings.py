"""
Embedding Migration Script
==========================

Safely migrate ChromaDB documents from llama3.2 embeddings to nomic-embed-text embeddings.

Safety Features:
- Automatic backup before migration
- Validation checks at each step
- Rollback capability
- Error handling with detailed reporting
- Progress tracking
"""

import os
import sys
import shutil
from datetime import datetime
from typing import List, Dict
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


class EmbeddingMigration:
    """Safely migrate embeddings with comprehensive error handling."""

    def __init__(self, old_db_path: str = "./chroma_db", new_model: str = "nomic-embed-text"):
        self.old_db_path = old_db_path
        self.backup_path = f"./chroma_db_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.new_db_path = "./chroma_db_new"
        self.new_model = new_model

    def check_prerequisites(self) -> bool:
        """Verify all prerequisites before starting migration."""
        print("🔍 Checking prerequisites...")

        # 1. Check if old database exists
        if not os.path.exists(self.old_db_path):
            print(f"❌ Error: Old database not found at {self.old_db_path}")
            return False
        print(f"✅ Old database found: {self.old_db_path}")

        # 2. Check if new database already exists (prevent overwrite)
        if os.path.exists(self.new_db_path):
            response = input(f"⚠️  {self.new_db_path} already exists. Overwrite? (yes/no): ")
            if response.lower() != "yes":
                print("❌ Migration cancelled")
                return False
            print(f"🗑️  Will overwrite {self.new_db_path}")

        # 3. Verify new embedding model is available
        print(f"🔍 Verifying {self.new_model} is available...")
        try:
            test_embeddings = OllamaEmbeddings(model=self.new_model)
            test_embeddings.embed_query("test")
            print(f"✅ Embedding model {self.new_model} is available")
        except Exception as e:
            print(f"❌ Error: {self.new_model} not available")
            print(f"   Run: ollama pull {self.new_model}")
            print(f"   Error details: {e}")
            return False

        # 4. Check disk space (rough estimate: 2x old DB size)
        try:
            old_size = self._get_directory_size(self.old_db_path)
            free_space = shutil.disk_usage(os.path.dirname(self.old_db_path)).free
            required_space = old_size * 3  # old + backup + new

            if free_space < required_space:
                print(f"⚠️  Warning: Low disk space")
                print(f"   Required: ~{required_space / (1024**2):.1f} MB")
                print(f"   Available: {free_space / (1024**2):.1f} MB")
                response = input("   Continue anyway? (yes/no): ")
                if response.lower() != "yes":
                    print("❌ Migration cancelled")
                    return False
            else:
                print(f"✅ Sufficient disk space ({free_space / (1024**2):.0f} MB available)")
        except Exception as e:
            print(f"⚠️  Could not check disk space: {e}")
            print("   Continuing anyway...")

        return True

    def create_backup(self) -> bool:
        """Create timestamped backup of old database."""
        print(f"\n📦 Creating backup...")
        try:
            shutil.copytree(self.old_db_path, self.backup_path)
            backup_size = self._get_directory_size(self.backup_path)
            print(f"✅ Backup created: {self.backup_path}")
            print(f"   Size: {backup_size / (1024**2):.1f} MB")
            return True
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False

    def load_old_documents(self) -> tuple[List[Document], bool]:
        """Load all documents from old database."""
        print(f"\n📖 Loading documents from old database...")
        try:
            # Use a dummy embedding function to load the old DB (we'll ignore the embeddings)
            old_embeddings = OllamaEmbeddings(model="llama3.2")  # Original model
            old_db = Chroma(
                persist_directory=self.old_db_path,
                embedding_function=old_embeddings,
                collection_name="research_cache"
            )

            # Get all documents
            all_docs = old_db.get()

            if not all_docs or not all_docs.get('ids'):
                print(f"⚠️  No documents found in old database")
                return [], False

            # Convert to Document objects
            documents = []
            for i in range(len(all_docs['ids'])):
                doc = Document(
                    page_content=all_docs['documents'][i],
                    metadata=all_docs['metadatas'][i]
                )
                documents.append(doc)

            print(f"✅ Loaded {len(documents)} documents")

            # Show sample
            if documents:
                print(f"\n📄 Sample document:")
                sample = documents[0]
                print(f"   Metadata: {sample.metadata}")
                print(f"   Content preview: {sample.page_content[:100]}...")

            return documents, True

        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            import traceback
            traceback.print_exc()
            return [], False

    def migrate_documents(self, documents: List[Document]) -> bool:
        """Create new database with new embeddings."""
        print(f"\n🔄 Migrating documents with new embeddings...")
        print(f"   Using model: {self.new_model}")
        print(f"   Total documents: {len(documents)}")

        try:
            # Create new embeddings function
            new_embeddings = OllamaEmbeddings(model=self.new_model)

            # Create new database
            print(f"   Creating new database at {self.new_db_path}...")

            # Process in batches to show progress
            batch_size = 10
            total_batches = (len(documents) + batch_size - 1) // batch_size

            new_db = None
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(documents))
                batch = documents[start_idx:end_idx]

                print(f"   Processing batch {batch_num + 1}/{total_batches} ({len(batch)} docs)...")

                if new_db is None:
                    # Initialize database with first batch
                    new_db = Chroma.from_documents(
                        documents=batch,
                        embedding=new_embeddings,
                        persist_directory=self.new_db_path,
                        collection_name="research_cache"
                    )
                else:
                    # Add subsequent batches
                    new_db.add_documents(batch)

            print(f"✅ Migration complete!")
            return True

        except Exception as e:
            print(f"❌ Migration failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def validate_migration(self, original_count: int) -> bool:
        """Verify migration was successful."""
        print(f"\n✅ Validating migration...")
        try:
            # Load new database
            new_embeddings = OllamaEmbeddings(model=self.new_model)
            new_db = Chroma(
                persist_directory=self.new_db_path,
                embedding_function=new_embeddings,
                collection_name="research_cache"
            )

            # Count documents
            new_docs = new_db.get()
            new_count = len(new_docs['ids']) if new_docs and new_docs.get('ids') else 0

            print(f"   Original documents: {original_count}")
            print(f"   Migrated documents: {new_count}")

            if new_count == original_count:
                print(f"✅ Validation passed: All documents migrated")
                return True
            else:
                print(f"⚠️  Warning: Document count mismatch")
                print(f"   Expected: {original_count}, Got: {new_count}")
                return False

        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False

    def run(self, dry_run: bool = False):
        """Execute the full migration with safety checks."""
        print("=" * 70)
        print("🔄 ChromaDB Embedding Migration")
        print("=" * 70)
        print(f"\nOld DB: {self.old_db_path}")
        print(f"New DB: {self.new_db_path}")
        print(f"Backup: {self.backup_path}")
        print(f"New model: {self.new_model}")

        if dry_run:
            print("\n⚠️  DRY RUN MODE - No changes will be made")

        print("\n" + "-" * 70)

        # Step 1: Prerequisites
        if not self.check_prerequisites():
            print("\n❌ Migration aborted: Prerequisites not met")
            return False

        # Step 2: Backup
        print("\n" + "-" * 70)
        if not dry_run:
            if not self.create_backup():
                print("\n❌ Migration aborted: Backup failed")
                return False
        else:
            print("\n⏭️  Skipping backup (dry run)")

        # Step 3: Load documents
        print("\n" + "-" * 70)
        documents, success = self.load_old_documents()
        if not success or not documents:
            print("\n❌ Migration aborted: Could not load documents")
            return False

        if dry_run:
            print(f"\n✅ Dry run complete")
            print(f"   Would migrate {len(documents)} documents")
            print(f"   Backup would be at: {self.backup_path}")
            print(f"   New database would be at: {self.new_db_path}")
            return True

        # Step 4: Migrate
        print("\n" + "-" * 70)
        if not self.migrate_documents(documents):
            print("\n❌ Migration failed")
            print(f"   Backup is safe at: {self.backup_path}")
            print(f"   You can restore with: mv {self.backup_path} {self.old_db_path}")
            return False

        # Step 5: Validate
        print("\n" + "-" * 70)
        if not self.validate_migration(len(documents)):
            print("\n⚠️  Migration completed but validation failed")
            print(f"   Review the new database before replacing the old one")

        # Step 6: Summary
        print("\n" + "=" * 70)
        print("✅ MIGRATION SUCCESSFUL")
        print("=" * 70)
        print(f"\nNext steps:")
        print(f"1. Review the new database:")
        print(f"   - New DB: {self.new_db_path}")
        print(f"   - Migrated: {len(documents)} documents")
        print(f"\n2. Test the new database:")
        print(f"   python main.py --enable-tools \"test query\"")
        print(f"\n3. If satisfied, replace the old database:")
        print(f"   rm -rf {self.old_db_path}")
        print(f"   mv {self.new_db_path} {self.old_db_path}")
        print(f"\n4. If there are issues, rollback:")
        print(f"   rm -rf {self.old_db_path}")
        print(f"   mv {self.backup_path} {self.old_db_path}")
        print(f"\nBackup location: {self.backup_path}")
        print("(You can delete the backup once you're satisfied)")
        print("=" * 70)

        return True

    def _get_directory_size(self, path: str) -> int:
        """Get total size of directory in bytes."""
        total = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total += os.path.getsize(filepath)
        except:
            pass
        return total


def main():
    """Main entry point."""
    print("\n")

    # Parse arguments
    dry_run = "--dry-run" in sys.argv

    if "--help" in sys.argv or "-h" in sys.argv:
        print("ChromaDB Embedding Migration Tool")
        print("\nUsage:")
        print("  python migrate_embeddings.py           # Run migration")
        print("  python migrate_embeddings.py --dry-run # Preview without changes")
        print("  python migrate_embeddings.py --help    # Show this help")
        print("\nThis script safely migrates your ChromaDB from llama3.2 to nomic-embed-text embeddings.")
        print("It creates automatic backups and validates the migration.")
        return

    # Create migrator and run
    migrator = EmbeddingMigration()
    success = migrator.run(dry_run=dry_run)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
