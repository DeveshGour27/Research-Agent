import sys

with open("app/db/repository.py", "r", encoding="utf-8") as f:
    content = f.read()

target = '''        stmt = (
            update(Job)
            .where(*where_conditions)
            .values(**values)
        )
        res = self.db.execute(stmt)
        self.db.commit()'''

replacement = '''        stmt = (
            update(Job)
            .where(*where_conditions)
            .values(**values)
        )
        res = self.db.execute(stmt)
        
        # --- PHASE 6 ASSISTANT MESSAGE CREATION ---
        if res.rowcount > 0 and status == "COMPLETED" and result is not None:
            from app.db.models import Message
            user_msg = self.db.execute(
                select(Message).where(Message.job_id == job_id, Message.role == "user")
            ).scalar_one_or_none()
            if user_msg:
                assistant_msg = Message(
                    chat_id=user_msg.chat_id,
                    role="assistant",
                    content=result,
                    job_id=job_id,
                )
                self.db.add(assistant_msg)

        self.db.commit()'''

if target in content:
    content = content.replace(target, replacement)
    with open("app/db/repository.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Patched successfully")
else:
    print("Target not found")
