import json
import unittest

from history_memory.events import EventStore

from ..observed_entity_slot import select_query_unique_observed_entity_slot


def tool(name, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {
                    field: {"type": "string"} for field in required
                },
                "required": list(required),
            },
        },
    }


class QueryUniqueObservedEntitySlotTest(unittest.TestCase):
    def setUp(self):
        self.tools = (
            tool("book_flight", ("access_token", "card_id")),
            tool("retrieve_invoice", ("access_token", "booking_id")),
        )

    @staticmethod
    def messages(booking_ids):
        rows = [{"role": "user", "content": "Book the flight."}]
        for index, booking_id in enumerate(booking_ids):
            call_id = f"call-{index}"
            rows.extend([
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "book_flight",
                            "arguments": json.dumps({
                                "access_token": "ABCDE12345",
                                "card_id": "1432",
                            }),
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps({
                        "booking_id": booking_id,
                        "transaction_id": f"tx-{index}",
                        "booking_status": True,
                        "booking_history": {"large": "ignored"},
                    }),
                },
            ])
        rows.extend([
            {
                "role": "assistant",
                "content": f"Booking ID: {booking_ids[-1]}",
            },
            {
                "role": "user",
                "content": (
                    "Fetch the invoice for that recent booking using "
                    "access_token=ABCDE12345."
                ),
            },
        ])
        return rows

    def test_selects_only_unique_query_required_booking_id(self):
        store = EventStore.from_messages("task-192", self.messages(["3426812"]))
        selection = select_query_unique_observed_entity_slot(store, self.tools)
        self.assertEqual(selection.status, "selected_unique_query_required_value")
        self.assertEqual(
            [(slot.field, slot.value) for slot in selection.slots],
            [("booking_id", "3426812")],
        )
        self.assertNotIn("transaction_id", [slot.field for slot in selection.slots])

    def test_ambiguous_visible_booking_ids_fail_closed(self):
        store = EventStore.from_messages(
            "ambiguous", self.messages(["3426812", "7654321"])
        )
        selection = select_query_unique_observed_entity_slot(store, self.tools)
        self.assertEqual(selection.status, "ambiguous_visible_values")
        self.assertEqual(selection.slots, ())


if __name__ == "__main__":
    unittest.main()
