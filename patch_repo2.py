import sys

with open("app/db/repository.py", "r", encoding="utf-8") as f:
    content = f.read()

new_methods = '''    def list_conversations(self, user_id: str):
        from app.db.models import Conversation
        from sqlalchemy import select
        stmt = select(Conversation).where(Conversation.user_id == user_id).order_by(Conversation.updated_at.desc())
        return list(self.db.execute(stmt).scalars().all())

    def create_conversation(self, user_id: str, title: str | None = None):
        from app.db.models import Conversation
        chat = Conversation(user_id=user_id, title=title)
        self.db.add(chat)
        self.db.commit()
        return chat

    def delete_conversation(self, chat_id: str, user_id: str) -> bool:
        from app.db.models import Conversation
        from sqlalchemy import delete
        stmt = delete(Conversation).where(Conversation.chat_id == chat_id, Conversation.user_id == user_id)
        res = self.db.execute(stmt)
        self.db.commit()
        return res.rowcount > 0

    def get_conversation(self'''

if "def get_conversation(self" in content and "def list_conversations" not in content:
    content = content.replace("    def get_conversation(self", new_methods)
    with open("app/db/repository.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Patched successfully")
else:
    print("Already patched or target not found")
