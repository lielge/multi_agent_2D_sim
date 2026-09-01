from __future__ import annotations

import os
import unittest


# Pygame must select the headless driver before it is imported or initialized.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

try:
    import pygame
except ModuleNotFoundError:  # The visualization dependency is optional.
    pygame = None  # type: ignore[assignment]
else:
    from multi_agent_sim.visualization.widgets import (
        Button,
        ChoiceSelector,
        IntegerSlider,
        TextInput,
    )


@unittest.skipUnless(pygame is not None, "pygame visualization extra is not installed")
class WidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.init()
        cls.surface = pygame.display.set_mode((360, 220))
        cls.font = pygame.font.Font(None, 20)

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.quit()

    def tearDown(self) -> None:
        pygame.key.set_mods(pygame.KMOD_NONE)
        self.surface.set_clip(None)
        self.surface.fill((0, 0, 0))

    def test_button_supports_mouse_keyboard_and_disabled_states(self) -> None:
        button = Button("Start", (20, 20, 100, 36), primary=True)

        self.assertTrue(
            button.handle_event(
                pygame.event.Event(
                    pygame.MOUSEBUTTONDOWN, button=1, pos=(40, 30)
                )
            )
        )
        self.assertTrue(button.focused)
        self.assertTrue(
            button.handle_event(
                pygame.event.Event(
                    pygame.KEYDOWN, key=pygame.K_RETURN, unicode="\r"
                )
            )
        )

        self.assertFalse(
            button.handle_event(
                pygame.event.Event(
                    pygame.MOUSEBUTTONDOWN, button=1, pos=(200, 200)
                )
            )
        )
        self.assertFalse(button.focused)
        self.assertFalse(
            button.handle_event(
                pygame.event.Event(
                    pygame.KEYDOWN, key=pygame.K_SPACE, unicode=" "
                )
            )
        )

        button.focused = True
        button.enabled = False
        self.assertFalse(button.focused)
        self.assertFalse(
            button.handle_event(
                pygame.event.Event(
                    pygame.KEYDOWN, key=pygame.K_RETURN, unicode="\r"
                )
            )
        )
        self.assertFalse(
            button.handle_event(
                pygame.event.Event(
                    pygame.MOUSEBUTTONDOWN, button=1, pos=(40, 30)
                )
            )
        )

        button.draw(self.surface, self.font, (40, 30))
        pygame.display.flip()

    def test_text_input_filters_length_and_edits_at_the_caret(self) -> None:
        field = TextInput(
            "12",
            (20, 20, 120, 32),
            max_length=3,
            character_filter=str.isdigit,
        )
        field.handle_event(
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(30, 30))
        )
        self.assertTrue(field.focused)

        self.assertTrue(
            field.handle_event(
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3, unicode="3")
            )
        )
        self.assertEqual(field.text, "123")
        self.assertFalse(
            field.handle_event(
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_4, unicode="4")
            )
        )
        self.assertFalse(
            field.handle_event(
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a, unicode="a")
            )
        )

        field.handle_event(
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT, unicode="")
        )
        self.assertTrue(
            field.handle_event(
                pygame.event.Event(
                    pygame.KEYDOWN, key=pygame.K_BACKSPACE, unicode=""
                )
            )
        )
        self.assertEqual(field.text, "13")
        self.assertTrue(
            field.handle_event(
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DELETE, unicode="")
            )
        )
        self.assertEqual(field.text, "1")

        field.handle_event(
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(300, 180))
        )
        self.assertFalse(field.focused)
        self.assertFalse(
            field.handle_event(
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_9, unicode="9")
            )
        )
        self.assertEqual(field.text, "1")

    def test_text_input_select_all_replaces_text_and_draw_restores_clip(self) -> None:
        field = TextInput(
            "123", (20, 20, 80, 32), max_length=4, character_filter=str.isdigit
        )
        field.set_text("123456")
        self.assertEqual(field.text, "1234")

        field.handle_event(
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(30, 30))
        )
        pygame.key.set_mods(pygame.KMOD_CTRL)
        self.assertFalse(
            field.handle_event(
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a, unicode="a")
            )
        )
        pygame.key.set_mods(pygame.KMOD_NONE)
        self.assertTrue(
            field.handle_event(
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_7, unicode="7")
            )
        )
        self.assertEqual(field.text, "7")

        original_clip = pygame.Rect(5, 5, 200, 100)
        self.surface.set_clip(original_clip)
        field.draw(self.surface, self.font, invalid=True)
        self.assertEqual(self.surface.get_clip(), original_clip)
        pygame.display.flip()

    def test_integer_slider_clamps_values_and_validates_range(self) -> None:
        with self.assertRaises(ValueError):
            IntegerSlider(1, 1, 1)
        with self.assertRaises(ValueError):
            IntegerSlider(5, 1, 3)

        slider = IntegerSlider(1, 30, 100, (20, 50, 290, 20))
        self.assertEqual(slider.value, 30)
        slider.value = -100
        self.assertEqual(slider.value, 1)

    def test_integer_slider_supports_dragging_and_keyboard_controls(self) -> None:
        slider = IntegerSlider(1, 30, 15, (20, 50, 290, 20))

        self.assertTrue(
            slider.handle_event(
                pygame.event.Event(
                    pygame.MOUSEBUTTONDOWN, button=1, pos=(20, 60)
                )
            )
        )
        self.assertTrue(slider.focused)
        self.assertTrue(slider.dragging)
        self.assertEqual(slider.value, 1)

        self.assertTrue(
            slider.handle_event(
                pygame.event.Event(
                    pygame.MOUSEMOTION, pos=(310, 60), rel=(290, 0), buttons=(1, 0, 0)
                )
            )
        )
        self.assertEqual(slider.value, 30)
        self.assertFalse(
            slider.handle_event(
                pygame.event.Event(
                    pygame.MOUSEBUTTONUP, button=1, pos=(310, 60)
                )
            )
        )
        self.assertFalse(slider.dragging)

        self.assertTrue(
            slider.handle_event(
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_HOME, unicode="")
            )
        )
        self.assertEqual(slider.value, 1)
        self.assertTrue(
            slider.handle_event(
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT, unicode="")
            )
        )
        self.assertEqual(slider.value, 2)
        self.assertTrue(
            slider.handle_event(
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_END, unicode="")
            )
        )
        self.assertEqual(slider.value, 30)
        self.assertFalse(
            slider.handle_event(
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP, unicode="")
            )
        )

        slider.draw(self.surface)
        pygame.display.flip()

        self.assertFalse(
            slider.handle_event(
                pygame.event.Event(
                    pygame.MOUSEBUTTONDOWN, button=1, pos=(350, 200)
                )
            )
        )
        self.assertFalse(slider.focused)
        self.assertFalse(slider.dragging)

    def test_choice_selector_cycles_with_mouse_and_keyboard(self) -> None:
        selector = ChoiceSelector(
            (("random", "Random"), ("nearest", "Nearest Item")),
            "random",
            (20, 20, 160, 32),
        )

        self.assertTrue(
            selector.handle_event(
                pygame.event.Event(
                    pygame.MOUSEBUTTONDOWN,
                    button=1,
                    pos=selector.rect.center,
                )
            )
        )
        self.assertEqual(selector.value, "nearest")
        self.assertEqual(selector.display_name, "Nearest Item")
        self.assertTrue(selector.focused)

        self.assertTrue(
            selector.handle_event(
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT, unicode="")
            )
        )
        self.assertEqual(selector.value, "random")
        selector.draw(self.surface, self.font, selector.rect.center)
        pygame.display.flip()

        self.assertFalse(
            selector.handle_event(
                pygame.event.Event(
                    pygame.MOUSEBUTTONDOWN,
                    button=1,
                    pos=(350, 200),
                )
            )
        )
        self.assertFalse(selector.focused)

    def test_choice_selector_validates_choices_and_value(self) -> None:
        with self.assertRaises(ValueError):
            ChoiceSelector(())
        with self.assertRaises(ValueError):
            ChoiceSelector((("same", "First"), ("same", "Second")))
        with self.assertRaises(ValueError):
            ChoiceSelector((("random", "Random"),), "missing")


if __name__ == "__main__":
    unittest.main()
